VPS setup for mcp-server-qdrant (Nginx Proxy Manager + n8n)
===========================================================

Goal: run mcp-server-qdrant on a VPS with Nginx Proxy Manager (NPM) and
streamable HTTP for n8n (`/mcp/` endpoint).

Prereqs
-------
- Docker + Docker Compose installed on the VPS
- Nginx Proxy Manager already running (network: `npm_default`)
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
EOF
```

3) Run container on the NPM network
-----------------------------------
```bash
docker rm -f mcp-qdrant 2>/dev/null
docker run -d --name mcp-qdrant \
  --network npm_default \
  --env-file .env \
  mcp-server-qdrant \
  mcp-server-qdrant --transport streamable-http
```

Verify from inside NPM:
```bash
docker exec -it npm_app_1 curl -i http://mcp-qdrant:8000/mcp/
```
Expected: 406 Not Acceptable unless you send `Accept: text/event-stream`.

4) Nginx Proxy Manager settings
-------------------------------
Create a Proxy Host:
- Domain: `qdrant-mcp.yourdomain.com`
- Forward Hostname/IP: `mcp-qdrant`
- Forward Port: `8000`
- Websockets: ON
- SSL: ON (Let's Encrypt)

5) Test over the domain
-----------------------
Initialize (creates a session):
```bash
curl -i -X POST https://qdrant-mcp.yourdomain.com/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

Use the `mcp-session-id` from the response:
```bash
curl -i https://qdrant-mcp.yourdomain.com/mcp/ \
  -H "Accept: text/event-stream" \
  -H "mcp-session-id: <PASTE_ID_HERE>"
```

6) n8n endpoint
--------------
Use:
```
https://qdrant-mcp.yourdomain.com/mcp/
```

Notes
-----
- The streamable HTTP endpoint is `/mcp/` (trailing slash matters).
- If you see `Bad Request: Missing session ID`, you need to run
  the initialize request first.
