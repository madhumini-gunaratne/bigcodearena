OG_ARENA_HARD_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."

JUDGE_SETTINGS = {
    "hard_prompt": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "coding": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "math": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "creative_writing": {
        "baseline": "gemini-2.0-flash-001",
        "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nWhen evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
    },
    "arena-hard-v0.1": {
        "baseline": "gpt-4-0314",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },

    "swe-arena": {
        "baseline": "gpt-4o",
        "system_prompt": """\
Please act as an impartial judge and evaluate the quality of the code provided by two AI assistants to the user prompt. You will be given assistant A's answer and assistant B's answer, along with the execution results of their code. Your job is to evaluate which assistant's generated code is better.

When evaluating the assistants' answers, compare both assistants'code execution results (e.g., stdout, stderr, and screenshot of the rendered code) first. You must identify and correct any mistakes or inaccurate information.

Note that the stderr may contain warnings only and you must not take it as an error. Due to the limitation of the execution environment, the errors may not be due to the code itself but the incompatibility issues or the lack of dependencies. These should be considered when evaluating the code.

There are several cases for the side-by-side comparison:
- Case 1: Both assistants' code execution results are successful. If screenshots are provided, you should compare the screenshots of the rendered code.
- Case 2: One assistant's code execution results are successful, while the other's are not. If the failure of the assistant's code execution results is due to the limitation of the execution environment, you MUST NOT penalize the assistant's response. You MUST carefully check the code generated by the assistant and judge the code correctness.
- Case 3: Both assistants' code execution results are not successful. You should compare both assistants' responses only. You MUST carefully check the code generated by the assistants and judge the code correctness. 

There are several scenarios for coding tasks:
- web development: the web page or application should be able to run in the browser and the user should be able to see the result. UI and UX are the most important factors.
- game development: the game should be able to run and the user should be able to see the result. UI, UX, and the game logic are the most important factors.
- creative coding: the artifact should produce a creative work. The creativity and novelty are the most important factors.
- problem solving: the code should be able to solve the problem described by the user. The correctness and efficiency are the most important factors.
- scientific computing: the code should use the proper scientific methods and tools to solve the problem. The correctness, efficiency, and visualization are the most important factors.
- diagram creation: the code should be able to create a diagram for logic or data flow. The visual presentation and the clarity are the most important factors.

YOU MUST IGNORE THE FAILURES OF THE CODE EXECUTION RESULTS THAT ARE DUE TO THE LIMITATION OF THE ENVIRONMENT. YOU MUST NOT JUDGE BASED ON THE EXISTENCE OF TEST CASES GENERATED BY THE ASSISTANTS. IF ANY SCREENSHOTS OR VISUAL OUTPUTS ARE PROVIDED, YOU MUST INSPECT THEM CAREFULLY FIRST. IF YOU CANNOT TELL THE QUALITY OF THE CODE BASED ON THE EXECUTION RESULTS, YOU SHOULD INSPECT THE CODE.

YOU MUST NOT TAKE THE COMPLEXITY OF THE SETUP PROCESS INTO ACCOUNT. REQUIRING MORE DEPENDENCIES DOES NOT MEAN THAT THE CODE IS LESS PREFERABLE. REMEMBER, THE OUTCOME IS MORE IMPORTANT THAN THE PROCESS. DEPENDENCIES DOES NOT MATTER.

THINK FROM THE USER'S PERSPECTIVE.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the similar or hard to tell: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".\
"""}
}