# Security

## API keys

- Store your OpenAI API key in `.secrets/openai.key`. This file is ignored by Git.
- Export the key via environment variables before running code:
  ```bash
  export OPENAI_API_KEY=$(cat .secrets/openai.key)
  # or point to a custom file
  export OPENAI_API_KEY_FILE=.secrets/openai.key
  ```
- Never commit API keys or other secrets to the repository.

## Git ignore protections

`.gitignore` already prevents secrets and logs from being tracked, including:

- `.secrets/`, `.env`, `*.env`, `*.key`
- `*.log`, `logs/`

These rules help keep credentials and runtime logs out of version control.
