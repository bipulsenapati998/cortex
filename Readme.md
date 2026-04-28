Open Git Bash terminal
docker ps | grep pgvector
docker restart d519f2842dbc
 # connect to postgres inside the container
docker compose exec pgvector psql -U cortex -d cortex_db
docker exec -it d519f2842dbc psql -U cortex -d cortex_db