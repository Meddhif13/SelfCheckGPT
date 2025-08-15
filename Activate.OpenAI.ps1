# Temporary policy for this session only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
# Activate venv, then set OPENAI key
. "$PSScriptRoot\.venv\Scripts\Activate.ps1"
. "$PSScriptRoot\.venv\Scripts\set-openai-env.ps1"
