#!/usr/bin/env python3
"""
Script to build all Docker images needed for sandbox execution.
This should be run once before executing any code to ensure all images are available.
"""

import os
import sys
import tempfile
import time
import subprocess
import shutil
from pathlib import Path

# Add the sandbox directory to the path
sys.path.append(str(Path(__file__).parent / "sandbox"))

from sandbox.docker_sandbox_manager import DOCKER_IMAGES, get_docker_client

def check_system_requirements():
    """Check system requirements for Docker builds"""
    print("🔍 Checking system requirements...")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # Check Docker daemon status
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            issues.append("❌ Docker daemon is not running or not accessible")
        else:
            print("✅ Docker daemon is running")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("❌ Docker command not found or timed out")
    
    # Check available disk space
    try:
        # Check current directory disk space
        current_dir = Path.cwd()
        disk_usage = shutil.disk_usage(current_dir)
        free_gb = disk_usage.free / (1024**3)
        total_gb = disk_usage.total / (1024**3)
        
        print(f"💾 Disk space: {free_gb:.1f}GB free out of {total_gb:.1f}GB total")
        
        if free_gb < 10:
            issues.append(f"❌ Insufficient disk space: only {free_gb:.1f}GB free (need at least 10GB)")
        elif free_gb < 20:
            warnings.append(f"⚠️  Low disk space: {free_gb:.1f}GB free (recommend at least 20GB)")
        else:
            print("✅ Sufficient disk space available")
            
    except Exception as e:
        warnings.append(f"⚠️  Could not check disk space: {e}")
    
    # Check available memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            lines = meminfo.split('\n')
            mem_total = 0
            mem_available = 0
            
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) // 1024  # Convert KB to MB
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) // 1024  # Convert KB to MB
                    break
            
            if mem_total > 0:
                print(f"🧠 Memory: {mem_available}MB available out of {mem_total}MB total")
                
                if mem_available < 2048:  # Less than 2GB
                    issues.append(f"❌ Insufficient memory: only {mem_available}MB available (need at least 2GB)")
                elif mem_available < 4096:  # Less than 4GB
                    warnings.append(f"⚠️  Low memory: {mem_available}MB available (recommend at least 4GB)")
                else:
                    print("✅ Sufficient memory available")
            else:
                warnings.append("⚠️  Could not determine memory information")
                
    except Exception as e:
        warnings.append(f"⚠️  Could not check memory: {e}")
    
    # Check Docker daemon resources
    try:
        client = get_docker_client()
        daemon_info = client.info()
        
        # Check Docker daemon memory limit
        if 'MemoryLimit' in daemon_info and daemon_info['MemoryLimit']:
            docker_mem_limit = daemon_info['MemoryLimit'] // (1024**3)  # Convert to GB
            print(f"🐳 Docker memory limit: {docker_mem_limit}GB")
            
            if docker_mem_limit < 2:
                warnings.append(f"⚠️  Docker memory limit is low: {docker_mem_limit}GB (recommend at least 2GB)")
            else:
                print("✅ Docker memory limit is sufficient")
        
        # Check Docker daemon storage driver
        if 'Driver' in daemon_info:
            storage_driver = daemon_info['Driver']
            print(f"🐳 Docker storage driver: {storage_driver}")
            
            if storage_driver in ['overlay2', 'overlay']:
                print("✅ Using modern overlay storage driver")
            else:
                warnings.append(f"⚠️  Using older storage driver: {storage_driver} (overlay2 recommended)")
        
        # Check Docker daemon version
        if 'ServerVersion' in daemon_info:
            server_version = daemon_info['ServerVersion']
            print(f"🐳 Docker daemon version: {server_version}")
            
            # Parse version to check if it's recent enough
            try:
                version_parts = server_version.split('.')
                major = int(version_parts[0])
                minor = int(version_parts[1])
                
                if major < 20 or (major == 20 and minor < 10):
                    warnings.append(f"⚠️  Docker daemon version {server_version} is older (20.10+ recommended)")
                else:
                    print("✅ Docker daemon version is recent")
            except:
                warnings.append("⚠️  Could not parse Docker daemon version")
                
    except Exception as e:
        issues.append(f"❌ Could not check Docker daemon: {e}")
    
    # Check network connectivity
    try:
        # Test Docker Hub connectivity
        result = subprocess.run(['docker', 'pull', 'hello-world:latest'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Network connectivity to Docker Hub is working")
            # Clean up test image
            subprocess.run(['docker', 'rmi', 'hello-world:latest'], 
                         capture_output=True, text=True, timeout=30)
        else:
            issues.append("❌ Cannot pull images from Docker Hub (network issue)")
    except subprocess.TimeoutExpired:
        issues.append("❌ Network connectivity test timed out (slow connection)")
    except Exception as e:
        warnings.append(f"⚠️  Could not test network connectivity: {e}")
    
    # Check Docker build context
    try:
        # Test creating a simple Dockerfile
        with tempfile.TemporaryDirectory() as test_context:
            test_dockerfile = os.path.join(test_context, "Dockerfile")
            with open(test_dockerfile, 'w') as f:
                f.write("FROM hello-world:latest\n")
            
            # Try a simple build
            result = subprocess.run(['docker', 'build', '-t', 'test-build', test_context], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ Docker build functionality is working")
                # Clean up test image
                subprocess.run(['docker', 'rmi', 'test-build'], 
                             capture_output=True, text=True, timeout=30)
            else:
                issues.append("❌ Docker build functionality is not working")
                if result.stderr:
                    issues.append(f"   Build error: {result.stderr.strip()}")
                    
    except subprocess.TimeoutExpired:
        issues.append("❌ Docker build test timed out (build system issue)")
    except Exception as e:
        warnings.append(f"⚠️  Could not test Docker build: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 System Check Summary:")
    
    if issues:
        print(f"\n❌ Critical Issues ({len(issues)}):")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Fix these issues before attempting to build Docker images.")
        return False
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"   {warning}")
        print("\n💡 These warnings may affect build performance but won't prevent builds.")
    
    if not issues and not warnings:
        print("\n🎉 All system checks passed! System is ready for Docker builds.")
    
    print(f"\n✅ System check completed with {len(issues)} issues and {len(warnings)} warnings")
    return len(issues) == 0

def format_time(seconds):
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"

def monitor_system_resources():
    """Monitor system resources during build"""
    try:
        # Check disk space
        current_dir = Path.cwd()
        disk_usage = shutil.disk_usage(current_dir)
        free_gb = disk_usage.free / (1024**3)
        
        # Check memory
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            lines = meminfo.split('\n')
            mem_available = 0
            
            for line in lines:
                if line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) // 1024  # Convert KB to MB
                    break
        
        return {
            'disk_free_gb': free_gb,
            'memory_available_mb': mem_available
        }
    except:
        return None

def check_build_health():
    """Check if system is healthy for building"""
    resources = monitor_system_resources()
    if not resources:
        return True  # Can't check, assume OK
    
    issues = []
    
    if resources['disk_free_gb'] < 5:
        issues.append(f"⚠️  Low disk space: {resources['disk_free_gb']:.1f}GB free")
    
    if resources['memory_available_mb'] < 1024:
        issues.append(f"⚠️  Low memory: {resources['memory_available_mb']}MB available")
    
    if issues:
        print("   🚨 System resource warnings:")
        for issue in issues:
            print(f"      {issue}")
    
    return len(issues) == 0

def build_docker_images(verbose=False, force=False):
    """Build all required Docker images"""
    print("🐳 Building Docker images for sandbox execution...")
    if verbose:
        print("🔍 Verbose mode enabled - showing detailed logs")
    if force:
        print("🔄 Force mode enabled - will rebuild existing images")
    print("=" * 60)
    
    client = get_docker_client()
    
    # Check Docker client version for compatibility
    try:
        import docker
        docker_version = docker.__version__
        print(f"🐳 Docker Python client version: {docker_version}")
    except:
        print("🐳 Docker Python client version: Unknown")
    
    # Check Docker daemon version
    try:
        daemon_info = client.info()
        docker_daemon_version = daemon_info.get('ServerVersion', 'Unknown')
        print(f"🐳 Docker daemon version: {docker_daemon_version}")
    except:
        print("🐳 Docker daemon version: Unknown")
    
    print("-" * 60)
    
    total_images = len(DOCKER_IMAGES)
    built_images = 0
    failed_images = 0
    total_start_time = time.time()
    
    for image_type, image_config in DOCKER_IMAGES.items():
        current_image_num = built_images + failed_images + 1
        print(f"\n📦 Building {image_type} image: {image_config.name}")
        print(f"   Progress: {current_image_num}/{total_images} ({(current_image_num/total_images)*100:.1f}%)")
        print("-" * 40)
        
        try:
            # Check if image already exists (unless force mode)
            if not force:
                try:
                    existing_image = client.images.get(image_config.name)
                    print(f"✅ Image {image_config.name} already exists (ID: {existing_image.id[:12]})")
                    print("   Skipping (use --force to rebuild)")
                    built_images += 1
                    continue
                except Exception:
                    pass
            else:
                # In force mode, remove existing image first
                try:
                    existing_image = client.images.get(image_config.name)
                    print(f"🔄 Removing existing image {image_config.name} (ID: {existing_image.id[:12]})")
                    client.images.remove(image_config.name, force=True)
                    print(f"   Removed existing image")
                except Exception:
                    pass
            
            # Create temporary directory for build context
            with tempfile.TemporaryDirectory() as build_context:
                dockerfile_path = os.path.join(build_context, "Dockerfile")
                
                # Write Dockerfile content
                with open(dockerfile_path, 'w') as f:
                    f.write(image_config.dockerfile_content)
                
                print(f"🔨 Building {image_config.name}...")
                print(f"   Started at: {time.strftime('%H:%M:%S')}")
                start_time = time.time()
                
                # Build the image with streaming logs
                build_success = False
                step_count = 0
                
                # Use the older API that's compatible with all Docker client versions
                try:
                    print("   🔨 Starting build process...")
                    
                    # Check system health before building
                    check_build_health()
                    
                    # Try with timeout first (newer versions)
                    try:
                        image, logs = client.images.build(
                            path=build_context,
                            tag=image_config.name,
                            rm=True,
                            forcerm=True,
                            timeout=600  # 10 minute timeout for build
                        )
                    except TypeError:
                        # Fallback for older versions without timeout support
                        print("   ⚠️  Using fallback build method (no timeout)")
                        image, logs = client.images.build(
                            path=build_context,
                            tag=image_config.name,
                            rm=True,
                            forcerm=True
                        )
                    
                    # Check system health after building
                    check_build_health()
                    
                    # Process logs after build completion
                    if logs:
                        print("   📝 Build completed. Processing logs:")
                        print("   " + "-" * 50)
                        
                        for log_entry in logs:
                            if isinstance(log_entry, dict):
                                if 'stream' in log_entry:
                                    log_line = log_entry['stream'].strip()
                                    if log_line:
                                        if log_line.startswith('Step '):
                                            step_count += 1
                                            print(f"   🔄 Step {step_count}: {log_line}")
                                        elif log_line.startswith(' --->'):
                                            print(f"   ✅ {log_line}")
                                        elif 'Pulling' in log_line:
                                            print(f"   📥 {log_line}")
                                        elif 'Verifying' in log_line:
                                            print(f"   🔍 {log_line}")
                                        elif 'Successfully' in log_line:
                                            print(f"   🎉 {log_line}")
                                            build_success = True
                                        elif 'error' in log_line.lower():
                                            print(f"   ❌ {log_line}")
                                        elif verbose or len(log_line) < 100:
                                            print(f"   📋 {log_line}")
                                        elif verbose:
                                            print(f"   📋 {log_line}")
                                
                                elif 'error' in log_entry:
                                    print(f"   ❌ Build error: {log_entry['error']}")
                                    raise Exception(f"Build failed: {log_entry['error']}")
                    
                    # If no logs or no success message, assume success
                    if not build_success:
                        build_success = True
                        print("   ✅ Build completed successfully")
                    
                except Exception as build_error:
                    print(f"   ❌ Build failed: {build_error}")
                    raise build_error
                
                build_time = time.time() - start_time
                print("   " + "-" * 50)
                print(f"✅ Successfully built {image_config.name} in {format_time(build_time)}")
                print(f"   Completed at: {time.strftime('%H:%M:%S')}")
                print(f"   Total steps: {step_count}")
                
                            # Get the built image to show its ID
            try:
                built_image = client.images.get(image_config.name)
                print(f"   Image ID: {built_image.id[:12]}")
            except Exception:
                print(f"   Image ID: Unable to retrieve")
            
            built_images += 1
                
        except Exception as e:
            print(f"❌ Failed to build {image_config.name}: {e}")
            print("   This will cause execution to fail for this image type")
            failed_images += 1
            continue
    
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("🎯 Docker image build process completed!")
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully built: {built_images}")
    print(f"   ❌ Failed to build: {failed_images}")
    print(f"   📦 Total images: {total_images}")
    print(f"   ⏱️  Total time: {format_time(total_time)}")
    
    # Check final status
    print(f"\n🔍 Final Image Status:")
    for image_type, image_config in DOCKER_IMAGES.items():
        try:
            existing_image = client.images.get(image_config.name)
            print(f"   ✅ {image_type}: {image_config.name} (ID: {existing_image.id[:12]})")
        except Exception:
            print(f"   ❌ {image_type}: {image_config.name} - BUILD FAILED")
    
    if failed_images > 0:
        print(f"\n⚠️  {failed_images} images failed to build.")
        print("   Fix the issues and run this script again.")
        print("   All images must be built successfully before running code execution.")
    else:
        print(f"\n🎉 All {total_images} images built successfully!")
        print("   You can now run code execution with 'python simple_execute.py'")
    
    return failed_images == 0

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Docker images for sandbox execution")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose build logs")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force rebuild existing images")
    parser.add_argument("--check-only", "-c", action="store_true",
                       help="Only check system requirements, don't build")
    parser.add_argument("--skip-check", "-s", action="store_true",
                       help="Skip system requirements check")
    
    args = parser.parse_args()
    
    try:
        # System check
        if not args.check_only and not args.skip_check:
            print("🔍 Performing system requirements check...")
            if not check_system_requirements():
                print("\n❌ System check failed. Please fix the issues above before building.")
                print("💡 Use --skip-check to bypass this check (not recommended).")
                sys.exit(1)
            print("\n" + "=" * 60)
        
        # If only checking, exit here
        if args.check_only:
            print("✅ System check completed. Use --skip-check to build images.")
            return
        
        # Build images
        build_docker_images(verbose=args.verbose, force=args.force)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Build process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
