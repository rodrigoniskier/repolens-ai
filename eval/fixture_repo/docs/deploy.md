# Deployment guide

## Docker

Build the container image with `docker build -t acme .` and run it with a mounted volume
for persistent data. The container listens on port 8000.

## Environment variables

Set `ACME_DATABASE_URL` to point at the production database and `ACME_LOG_LEVEL` to control
verbosity. Never commit secrets; inject them at deploy time.

## Rollbacks

To roll back a release, redeploy the previous image tag and run the database downgrade script.
