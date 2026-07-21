# Deploying

`example.com.conf` is an nginx site template: it serves the static pages from
`web/` and proxies `/sudoku-api/` and `/socket.io/` to the Sudoku backend.
Replace `example.com` with your own domain — it appears in `server_name`, the
document root and the Let's Encrypt certificate paths — then:

```bash
# static pages
cd web
scp -r ./* root@YOUR_HOST:/var/www/example.com/
```

```bash
# sudoku multiplayer backend
cd back/sudoku-back/
scp server.js rooms.js gameState.js package.json root@YOUR_HOST:/var/www/sudoku-backend/
```

```bash
# nginx site
cd deploy
scp example.com.conf root@YOUR_HOST:/etc/nginx/sites-available/
```

The backend reads its allowed browser origins from `ALLOWED_ORIGINS`
(comma-separated, defaults to `http://localhost:8080`):

```bash
ALLOWED_ORIGINS=https://example.com node server.js
```
