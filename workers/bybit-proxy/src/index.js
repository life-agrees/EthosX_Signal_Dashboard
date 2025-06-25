addEventListener('fetch', e => e.respondWith(handle(e.request)))

async function handle(request) {
  const url = new URL(request.url)
  const target = `https://api.bybit.com${url.pathname}${url.search}`
  const resp = await fetch(target, { method: request.method })
  const body = await resp.text()
  return new Response(body, {
    status:  resp.status,
    headers: resp.headers
  })
}
