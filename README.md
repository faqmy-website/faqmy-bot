# bot-demo

## Project setup

```
# Launch elastic
docker compose up -d

# Install dependencies
poetry install

# Setup env variables
cp .evn.sample .env
nano .env

# Populate index
poetry run python build_index.py

# Ask bot a question.
poetry run python bot_query.py
```
