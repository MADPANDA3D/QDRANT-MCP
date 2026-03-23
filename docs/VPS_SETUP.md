VPS setup for mcp-server-qdrant (Nginx Proxy Manager + n8n)
===========================================================

Goal: run mcp-server-qdrant on a VPS with Nginx Proxy Manager (NPM) and
streamable HTTP for n8n (`/mcp` canonical endpoint).

Prereqs
-------
- Docker + Docker Compose installed on the VPS
- Nginx Proxy Manager already running (network: `mcp-network` (shared with NPM))
- Qdrant Cloud URL + API key

1) Clone and build
------------------
```bash
mkdir -p ~/qdrant-mcp
cd ~/qdrant-mcp
git clone https://github.com/qdrant/mcp-server-qdrant.git
cd mcp-server-qdrant
docker build -t mcp-server-qdrant .
```

2) Create .env (no quotes, one line per key)
--------------------------------------------
```bash
cat > .env <<'EOF'
QDRANT_URL=https://YOUR-QDRANT-ID.YOUR-REGION.cloud.qdrant.io:6333
QDRANT_API_KEY=YOUR_API_KEY
COLLECTION_NAME=jarvis-knowledge-base
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# For OpenAI embeddings (example):
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-large
# OPENAI_API_KEY=your_openai_key
FASTMCP_SERVER_HOST=0.0.0.0
FASTMCP_SERVER_PORT=8000
# Trust forwarded proto/client headers only from your shared proxy network
FORWARDED_ALLOW_IPS=172.22.0.0/16
EOF
```

3) Run container on the NPM network
-----------------------------------
```bash
docker rm -f mcp-qdrant 2>/dev/null
docker run -d --name mcp-qdrant \
  --network mcp-network \
  --env-file .env \
  mcp-server-qdrant \
  mcp-server-qdrant --transport streamable-http
```

Verify from inside NPM:
```bash
docker exec -it npm_app_1 curl -i http://mcp-qdrant:8000/mcp/
```
Expected: 406 Not Acceptable unless you send `Accept: text/event-stream`.

3b) Optional auto-update with Watchtower (GHCR)
-----------------------------------------------
If you prefer auto-updates on each release, run the GHCR image instead of a local
build. Make sure the GHCR package is public (or login to GHCR first).

```bash
docker rm -f mcp-qdrant 2>/dev/null
docker run -d --name mcp-qdrant \
  --network mcp-network \
  --env-file .env \
  --label com.centurylinklabs.watchtower.enable=true \
  ghcr.io/<owner>/mad-mcp-qdrant:latest \
  mcp-server-qdrant --transport streamable-http

docker run -d --name watchtower \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower \
  --cleanup --interval 1800 --label-enable
```

4) Nginx Proxy Manager settings
-------------------------------
Create a Proxy Host:
- Domain: `qdrant-mcp.yourdomain.com`
- Forward Hostname/IP: `mcp-qdrant`
- Forward Port: `8000`
- Websockets: ON
- SSL: ON (Let's Encrypt)

Advanced (exact path behavior):
```nginx
location = /mcp {
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection $http_connection;
  proxy_http_version 1.1;
  proxy_set_header Host $host;
  proxy_set_header X-Forwarded-Scheme $scheme;
  proxy_set_header X-Forwarded-Proto $scheme;
  proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  proxy_set_header X-Real-IP $remote_addr;
  proxy_buffering off;
  proxy_read_timeout 3600s;
  proxy_pass http://mcp-qdrant:8000/mcp/;
}

location = /mcp/ {
  default_type application/json;
  add_header Cache-Control "no-store";
  return 410 '{"error":"deprecated_endpoint","message":"Deprecated MCP URL. Use https://qdrant-mcp.yourdomain.com/mcp (remove trailing slash)."}';
}
```

5) Test over the domain
-----------------------
Initialize (creates a session):
```bash
curl -i -X POST https://qdrant-mcp.yourdomain.com/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

Use the `mcp-session-id` from the response:
```bash
curl -i https://qdrant-mcp.yourdomain.com/mcp \
  -H "Accept: text/event-stream" \
  -H "mcp-session-id: <PASTE_ID_HERE>"
```

6) n8n endpoint
--------------
Use:
```
https://qdrant-mcp.yourdomain.com/mcp
```

Notes
-----
- Canonical public endpoint is `/mcp` (no trailing slash).
- Deprecated endpoint `/mcp/` should return `410 Gone` with a migration message.
- If you see `Bad Request: Missing session ID`, you need to run
  the initialize request first.
