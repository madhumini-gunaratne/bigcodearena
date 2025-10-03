'''
Constants for sandbox.
'''

from enum import StrEnum
import os

from .prompts import (
    DEFAULT_C_CODE_RUN_SANDBOX_INSTRUCTION, DEFAULT_CPP_CODE_RUN_SANDBOX_INSTRUCTION, DEFAULT_GOLANG_CODE_RUN_SANDBOX_INSTRUCTION, DEFAULT_GRADIO_SANDBOX_INSTRUCTION, DEFAULT_HTML_SANDBOX_INSTRUCTION, DEFAULT_JAVA_CODE_RUN_SANDBOX_INSTRUCTION, DEFAULT_JAVASCRIPT_RUNNER_INSTRUCTION, DEFAULT_MERMAID_SANDBOX_INSTRUCTION, DEFAULT_PYGAME_SANDBOX_INSTRUCTION, DEFAULT_PYTHON_RUNNER_INSTRUCTION, DEFAULT_REACT_SANDBOX_INSTRUCTION, DEFAULT_RUST_CODE_RUN_SANDBOX_INSTRUCTION, DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION, DEFAULT_VUE_SANDBOX_INSTRUCTION, GENERAL_SANDBOX_INSTRUCTION
)

# Import Python configuration
try:
    from .python_config import PYTHON_EXECUTABLE, PYTHON_HTTP_SERVER_MODULE, PIP_EXECUTABLE
except ImportError:
    # Fallback if the config module is not available
    PYTHON_EXECUTABLE = os.environ.get("PYTHON_EXECUTABLE", "python3.10")
    PYTHON_HTTP_SERVER_MODULE = f"{PYTHON_EXECUTABLE} -m http.server"
    PIP_EXECUTABLE = f"{PYTHON_EXECUTABLE} -m pip"

E2B_API_KEY = os.environ.get("E2B_API_KEY")
'''
API key for the e2b API.
'''

AZURE_BLOB_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
'''
API key for the Azure Blob Storage.
'''

AZURE_BLOB_STORAGE_CONTAINER_NAME = "softwarearenalogs"
'''
Contianer name for the Azure Blob Storage.
'''

SANDBOX_TEMPLATE_ID: str = "bxq9sha9l55ytsyfturr"
'''
Template ID for the sandbox.
'''

SANDBOX_NGINX_PORT: int = 8000
'''
Nginx port for the sandbox.
'''

SANDBOX_TIMEOUT_SECONDS: int = 30
'''
Timeout in seconds for created sandboxes to expire.
'''

CODE_RUN_TIMEOUT_SECONDS: int = 60
'''
Timeout in seconds for code execution.
'''

SANDBOX_RETRY_COUNT: int = 3
'''
Number of times to retry the sandbox creation.
'''

INSTALLED_PYPI_PACKAGES = [
    "boto3",
    "botocore",
    "urllib3",
    "setuptools",
    "requests",
    "certifi",
    "idna",
    "charset-normalizer",
    "packaging",
    "typing-extensions",
    "python-dateutil",
    "aiobotocore",
    "s3transfer",
    "grpcio-status",
    "pyyaml",
    "six",
    "fsspec",
    "s3fs",
    "numpy",
    "wheel",
    "pip",
    "cryptography",
    "awscli",
    "pydantic",
    "cffi",
    "attrs",
    "google-api-core",
    "pycparser",
    "pandas",
    "importlib-metadata",
    "jmespath",
    "click",
    "zipp",
    "rsa",
    "pyasn1",
    "markupsafe",
    "pytz",
    "colorama",
    "protobuf",
    "platformdirs",
    "jinja2",
    "rich",
    "tomli",
    "pytest",
    "pydantic-core",
    "pyjwt",
    "pluggy",
    "aiohttp",
    "virtualenv",
    "jsonschema",
    "googleapis-common-protos",
    "cachetools",
    "google-auth",
    "filelock",
    "wrapt",
    "sqlalchemy",
    "docutils",
    "pyasn1-modules",
    "pyarrow",
    "greenlet",
    "iniconfig",
    "pygments",
    "annotated-types",
    "yarl",
    "requests-oauthlib",
    "tzdata",
    "psutil",
    "multidict",
    "pyparsing",
    "requests-toolbelt",
    "exceptiongroup",
    "werkzeug",
    "soupsieve",
    "oauthlib",
    "beautifulsoup4",
    "frozenlist",
    "more-itertools",
    "distlib",
    "tomlkit",
    "pathspec",
    "aiosignal",
    "grpcio",
    "tqdm",
    "scipy",
    "async-timeout",
    "pillow",
    "isodate",
    "anyio",
    "sortedcontainers",
    "decorator",
    "markdown-it-py",
    "deprecated",
    "mypy-extensions",
    "sniffio",
    "httpx",
    "coverage",
    "openpyxl",
    "flask",
    "rpds-py",
    "et-xmlfile"
]


class SandboxEnvironment(StrEnum):
    AUTO = 'Auto'

    # Web UI Frameworks
    HTML = 'HTML'
    REACT = 'React'
    VUE = 'Vue'
    GRADIO = 'Gradio'
    STREAMLIT = 'Streamlit'
    PYGAME = 'PyGame'
    MERMAID = 'Mermaid'

    # Runner
    PYTHON_RUNNER = 'Python Runner'
    JAVASCRIPT_RUNNER = 'Javascript Runner'

    # Compiler
    C_RUNNER = 'C Runner'
    CPP_RUNNER = 'C++ Runner'
    # CSHARP_RUNNER = 'C# Runner'
    JAVA_RUNNER = 'Java Runner'
    RUST_RUNNER = 'Rust Runner'
    GOLANG_RUNNER = 'Golang Runner'


DEFAULT_SANDBOX_INSTRUCTIONS: dict[SandboxEnvironment, str] = {
    SandboxEnvironment.AUTO: GENERAL_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.PYTHON_RUNNER: DEFAULT_PYTHON_RUNNER_INSTRUCTION.strip(),
    SandboxEnvironment.JAVASCRIPT_RUNNER: DEFAULT_JAVASCRIPT_RUNNER_INSTRUCTION.strip(),
    SandboxEnvironment.HTML: DEFAULT_HTML_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.REACT: DEFAULT_REACT_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.VUE: DEFAULT_VUE_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.GRADIO: DEFAULT_GRADIO_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.STREAMLIT: DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.PYGAME: DEFAULT_PYGAME_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.MERMAID: DEFAULT_MERMAID_SANDBOX_INSTRUCTION.strip(),
    # Runners
    SandboxEnvironment.C_RUNNER: DEFAULT_C_CODE_RUN_SANDBOX_INSTRUCTION,
    SandboxEnvironment.CPP_RUNNER: DEFAULT_CPP_CODE_RUN_SANDBOX_INSTRUCTION,
    SandboxEnvironment.JAVA_RUNNER: DEFAULT_JAVA_CODE_RUN_SANDBOX_INSTRUCTION,
    SandboxEnvironment.GOLANG_RUNNER: DEFAULT_GOLANG_CODE_RUN_SANDBOX_INSTRUCTION,
    SandboxEnvironment.RUST_RUNNER: DEFAULT_RUST_CODE_RUN_SANDBOX_INSTRUCTION,

}
