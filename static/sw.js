// Minimal service worker: makes the web UI an installable PWA and lets the
// app shell load offline. Deliberately conservative — it never touches POST
// requests or /api/* so the live debug UI keeps working against the network.
const CACHE = "va-shell-v1";
const SHELL = [
  "/",
  "/manifest.webmanifest",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/static/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return; // never intercept POST (/api/send, /api/dry-run, ...)
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return; // dynamic data: always go to network

  if (req.mode === "navigate") {
    // App shell: network-first so the UI stays fresh, fall back to cache offline.
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put("/", copy));
          return res;
        })
        .catch(() => caches.match("/"))
    );
    return;
  }

  // Static assets (icons, manifest): cache-first.
  event.respondWith(caches.match(req).then((res) => res || fetch(req)));
});
