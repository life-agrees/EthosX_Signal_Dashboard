addEventListener('fetch', e => e.respondWith(handle(e.request)))

async function handle(request) {
  // Handle CORS preflight requests
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Max-Age': '86400',
      }
    })
  }

  const url = new URL(request.url)
  
  // If it's a Bybit API path, proxy to Bybit
  if (url.pathname.startsWith('/v5/')) {
    const target = `https://api.bybit.com${url.pathname}${url.search}`
    console.log(`Proxying to Bybit: ${target}`)
    
    try {
      const resp = await fetch(target, { 
        method: request.method,
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; CloudflareWorker/1.0)',
          'Accept': 'application/json'
        }
      })
      
      const body = await resp.text()
      
      return new Response(body, {
        status: resp.status,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        }
      })
    } catch (error) {
      return new Response(JSON.stringify({
        error: 'Failed to fetch from Bybit API',
        details: error.message
      }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      })
    }
  }
  
  // Everything else goes to your backend
  else {
    const target = `https://ethosx-signal-dashboard.onrender.com${url.pathname}${url.search}`
    console.log(`Proxying to Backend: ${target}`)
    
    try {
      const resp = await fetch(target, { 
        method: request.method,
        headers: request.headers
      })
      
      const body = await resp.text()
      
      return new Response(body, {
        status: resp.status,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        }
      })
    } catch (error) {
      return new Response(JSON.stringify({
        error: 'Failed to fetch from backend',
        details: error.message
      }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      })
    }
  }
}