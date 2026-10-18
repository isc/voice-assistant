// Minimal service worker: makes the web UI an installable PWA, lets the
// app shell load offline, and shows a timer's push. Deliberately conservative — it never touches POST
// requests or api/* so the live UI keeps working against the network.
// URLs are relative to this script: the app lives at / on the tailnet, at
// /assistant/ through Funnel.
const CACHE = "va-shell-v4";
const SHELL = [
  "./",
  "manifest.webmanifest",
  "static/icon-192.png",
  "static/icon-512.png",
  "static/apple-touch-icon.png",
];
const API = new URL("api/", self.location).pathname;

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
  if (req.method !== "GET") return; // never intercept POST (api/send, api/dry-run, ...)
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith(API)) return; // dynamic data: always go to network

  if (req.mode === "navigate") {
    // App shell: network-first so the UI stays fresh, fall back to cache offline.
    // Only a page that loaded is kept: not the key link's redirect, nor a 401.
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put("./", copy));
          }
          return res;
        })
        .catch(() => caches.match("./"))
    );
    return;
  }

  // Static assets (icons, manifest): cache-first.
  event.respondWith(caches.match(req).then((res) => res || fetch(req)));
});

// A timer set from this phone rings (AD-070): the server's push, shown whether the app is open or not (a push
// must show something). Silent while the app is shown: the page has chimed
self.addEventListener("push", (event) => {
  const message = event.data.json();
  event.waitUntil(
    self.clients.matchAll({ type: "window" }).then((open) => {
      const shown = open.some((c) => c.visibilityState === "visible");
      return self.registration.showNotification(message.title, {
        body: message.body,
        tag: message.tag,
        renotify: true,
        icon: "static/icon-192.png",
        lang: "fr",
        ...(shown ? { silent: true } : { requireInteraction: true, vibrate: [300, 150, 300, 150, 600] }),
      });
    })
  );
});

// A tap on it brings the app forward, or opens it
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(
    self.clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((open) => (open.length ? open[0].focus() : self.clients.openWindow("./")))
  );
});
