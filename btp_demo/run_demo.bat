@echo off
rem Live demo: one LLM request (Groq, GPT-OSS 120B), then run the rule on the 6-job instance.
rem   run_demo.bat           live call
rem   run_demo.bat --saved   no internet needed: Gemma's rule from the report
cd /d "%~dp0"
..\Code\.venv\Scripts\python.exe demo.py %*
