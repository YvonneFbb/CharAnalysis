(function (window) {
  const DEFAULT_API_BASE = '/api';
  const DEFAULT_LOCALHOST_ORIGIN = 'http://127.0.0.1:5001';

  function toApiBase(origin) {
    return `${String(origin || '').replace(/\/$/, '')}/api`;
  }

  function apiBases(options = {}) {
    const bases = [];
    if (options.preferRelative !== false) {
      bases.push(DEFAULT_API_BASE);
    }

    const origins = [];
    const ownOrigin = window.location.origin;
    if (ownOrigin && ownOrigin !== 'null') {
      origins.push(ownOrigin);
    }

    if (options.includeParent !== false) {
      try {
        const parentOrigin = window.top?.location?.origin;
        if (parentOrigin && parentOrigin !== 'null') {
          origins.push(parentOrigin);
        }
      } catch (_) {}
    }

    if (options.includeLocalhost) {
      origins.push(options.localhostOrigin || DEFAULT_LOCALHOST_ORIGIN);
    }

    for (const origin of origins) {
      bases.push(toApiBase(origin));
    }
    return Array.from(new Set(bases));
  }

  function apiBase() {
    return (window.REVIEW_API_BASE || apiBases()[0] || DEFAULT_API_BASE).replace(/\/$/, '');
  }

  function url(path, params) {
    let raw = String(path || '');
    if (!/^https?:\/\//.test(raw) && !raw.startsWith('/api')) {
      raw = `${apiBase()}${raw.startsWith('/') ? raw : `/${raw}`}`;
    }
    if (!params) return raw;

    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== '') {
        query.set(key, value);
      }
    }
    const suffix = query.toString();
    if (!suffix) return raw;
    return `${raw}${raw.includes('?') ? '&' : '?'}${suffix}`;
  }

  async function parseResponse(response) {
    const contentType = (response.headers.get('content-type') || '').toLowerCase();
    const payload = contentType.includes('application/json')
      ? await response.json()
      : { success: response.ok, text: await response.text() };
    if (!response.ok) {
      const message = payload && (payload.error || payload.text);
      throw new Error(message || `HTTP ${response.status}`);
    }
    return payload;
  }

  async function getJSON(path, params) {
    return parseResponse(await fetch(url(path, params)));
  }

  async function postJSON(path, body) {
    return parseResponse(await fetch(url(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    }));
  }

  window.ReviewApi = { apiBase, apiBases, url, getJSON, postJSON };
})(window);
