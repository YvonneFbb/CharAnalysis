(function (window) {
  function createStore(options) {
    const {
      apiBase,
      apiBases,
      debug,
      getReviewResults,
      setReviewResults,
      getDirtyBooks,
      getDirtyCharsByBook,
      clearDirtyState,
      setSyncStatus,
      persistReviewResults,
    } = options;

    let saveInFlight = false;
    let saveQueued = false;

    function ensureReviewResultsShape() {
      const current = getReviewResults();
      if (!current.version || current.version !== 2) {
        current.version = 2;
      }
      if (!current.books || typeof current.books !== 'object') {
        current.books = {};
      }
      return current;
    }

    function validateDataFormat(data) {
      if (!data || typeof data !== 'object') {
        return { valid: false, error: '数据不是有效对象' };
      }

      if (data.version !== 2) {
        return { valid: false, error: `不支持的数据版本: ${data.version}，期望版本 2` };
      }

      if (!data.books || typeof data.books !== 'object') {
        return { valid: false, error: '缺少 books 字段' };
      }

      const utcPattern = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;

      for (const bookName in data.books) {
        const book = data.books[bookName];
        for (const char in book) {
          const charData = book[char];

          if (!charData || typeof charData !== 'object') {
            return { valid: false, error: `无效的字符数据: ${bookName} - ${char}` };
          }

          if (!charData.instances || typeof charData.instances !== 'object') {
            return { valid: false, error: `缺少 instances 字段: ${bookName} - ${char}` };
          }

          if (!charData.timestamp) {
            return { valid: false, error: `缺少 timestamp 字段: ${bookName} - ${char}` };
          }

          if (!utcPattern.test(charData.timestamp)) {
            return {
              valid: false,
              error: `无效的时间戳格式: ${bookName} - ${char}: ${charData.timestamp}（必须是 UTC 格式，如 "2025-10-27T13:50:12.804Z"）`
            };
          }
        }
      }

      return { valid: true };
    }

    function initializeResults() {
      setReviewResults({ version: 2, books: {} });
    }

    function mergeResults(serverResults) {
      const reviewResults = ensureReviewResultsShape();
      const serverData = serverResults.books;

      for (const bookName in serverData) {
        const serverBook = serverData[bookName];

        if (!reviewResults.books[bookName]) {
          reviewResults.books[bookName] = serverBook;
          continue;
        }

        const localBook = reviewResults.books[bookName];

        for (const char in serverBook) {
          const serverCharData = serverBook[char];
          const localCharData = localBook[char];
          const serverInstanceCount = Object.keys(serverCharData.instances || {}).length;

          if (!localCharData) {
            localBook[char] = serverCharData;
            continue;
          }

          const serverTime = new Date(serverCharData.timestamp || 0).getTime();
          const localTime = new Date(localCharData.timestamp || 0).getTime();
          const localInstanceCount = Object.keys(localCharData.instances || {}).length;

          if (serverTime > localTime || serverInstanceCount > localInstanceCount) {
            localBook[char] = serverCharData;
          }
        }
      }

      persistReviewResults();
    }

    async function syncBookFromServer(bookName) {
      try {
        const response = await fetch(`${apiBase}/load_review?book=${encodeURIComponent(bookName)}`);
        const data = await response.json();

        if (!data.success || !data.data) return;

        const validation = validateDataFormat(data.data);
        if (!validation.valid) {
          console.error('❌ 服务器数据格式错误:', validation.error);
          return;
        }

        const reviewResults = ensureReviewResultsShape();
        const serverBook = data.data.books?.[bookName];
        reviewResults.books[bookName] = serverBook || {};
      } catch (error) {
        console.error(`同步书籍 "${bookName}" 失败:`, error);
      }
    }

    function buildDirtyPayload() {
      const reviewResults = ensureReviewResultsShape();
      const dirtyBooks = getDirtyBooks();
      const dirtyCharsByBook = getDirtyCharsByBook();

      if (Object.keys(reviewResults.books).length === 0 || dirtyBooks.size === 0) {
        return null;
      }

      const payload = { version: 2, books: {} };
      for (const bookName of dirtyBooks) {
        if (!reviewResults.books[bookName]) continue;
        const dirtyChars = dirtyCharsByBook.get(bookName);
        if (dirtyChars && dirtyChars.size > 0) {
          const subset = {};
          dirtyChars.forEach(char => {
            if (reviewResults.books[bookName][char]) {
              subset[char] = reviewResults.books[bookName][char];
            }
          });
          payload.books[bookName] = subset;
        } else {
          payload.books[bookName] = reviewResults.books[bookName];
        }
      }

      return Object.keys(payload.books).length > 0 ? payload : null;
    }

    function sendBeaconPayload(payload) {
      if (!payload) return false;
      const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
      try {
        return navigator.sendBeacon(`${apiBase}/save_review`, blob);
      } catch (_) {
        return false;
      }
    }

    async function fetchWithTimeout(url, options, timeoutMs) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      try {
        return await fetch(url, { ...options, signal: controller.signal });
      } finally {
        clearTimeout(timeoutId);
      }
    }

    function sendViaXHR(url, body) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url, true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onload = () => resolve({
          ok: xhr.status >= 200 && xhr.status < 300,
          status: xhr.status,
          text: xhr.responseText || ''
        });
        xhr.onerror = () => reject(new Error('xhr error'));
        xhr.send(body);
      });
    }

    async function fetchWithFallback(path, options, timeoutMs) {
      let lastErr = null;
      for (const base of apiBases) {
        try {
          if (debug) {
            console.log('[OCR] save_review try:', `${base}${path}`);
          }
          return await fetchWithTimeout(`${base}${path}`, options, timeoutMs);
        } catch (error) {
          if (debug) {
            console.warn('[OCR] save_review error:', error);
          }
          lastErr = error;
        }
      }
      throw lastErr || new Error('fetch failed');
    }

    async function saveToServer() {
      if (saveInFlight) {
        saveQueued = true;
        return;
      }

      const payload = buildDirtyPayload();
      if (!payload) {
        return;
      }

      saveInFlight = true;
      const jsonStr = JSON.stringify(payload);

      try {
        if (debug) {
          console.log('[OCR] save_review payload size KB:', (jsonStr.length / 1024).toFixed(2));
          console.log('[OCR] save_review books:', Object.keys(payload.books));
        }
        setSyncStatus('正在保存…');

        const response = await fetchWithFallback('/save_review', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: jsonStr
        }, 30000);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        if (data.success) {
          clearDirtyState();
          setSyncStatus('已保存');
        } else {
          console.error('✗ 保存到服务器失败:', data.error);
          setSyncStatus('保存失败：' + (data.error || '未知错误'), true);
        }
      } catch (error) {
        if (error.name === 'AbortError') {
          console.error('✗ 保存到服务器超时（30秒）');
        } else {
          console.error('✗ 保存到服务器失败:', error.message);
        }
        console.warn('[OCR] save_review failed context:', {
          href: window.location.href,
          origin: window.location.origin,
          apiBases
        });
        setSyncStatus('保存失败：' + (error.message || '网络错误'), true);

        if (sendBeaconPayload(payload)) {
          clearDirtyState();
          setSyncStatus('已保存');
        } else {
          try {
            for (const base of apiBases) {
              const xhrResp = await sendViaXHR(`${base}/save_review`, jsonStr);
              if (xhrResp.ok) {
                clearDirtyState();
                setSyncStatus('已保存');
                break;
              }
            }
          } catch (_) {}
        }
      } finally {
        saveInFlight = false;
        if (saveQueued) {
          saveQueued = false;
          saveToServer();
        }
      }
    }

    return {
      initializeResults,
      validateDataFormat,
      mergeResults,
      syncBookFromServer,
      buildDirtyPayload,
      sendBeaconPayload,
      saveToServer,
    };
  }

  window.OcrReviewStore = { createStore };
})(window);
