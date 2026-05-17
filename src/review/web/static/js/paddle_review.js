(function () {
  const { getJSON, postJSON } = window.ReviewApi;
  const { qs, clear, text, el, option } = window.ReviewDom;

  const state = {
    currentBook: null,
    currentChar: null,
    charItems: [],
    presetBook: new URLSearchParams(window.location.search).get('book'),
    presetChar: new URLSearchParams(window.location.search).get('char'),
  };

  async function loadBooks() {
    const select = qs('#book-select');
    clear(select);
    const data = await getJSON('/paddle/books');
    if (!data.success) return;

    const fragment = document.createDocumentFragment();
    for (const book of data.books || []) {
      fragment.appendChild(option(book.name, book.name));
    }
    select.appendChild(fragment);

    if ((data.books || []).length > 0) {
      const hasPreset = state.presetBook && data.books.some(book => book.name === state.presetBook);
      state.currentBook = hasPreset ? state.presetBook : data.books[0].name;
      select.value = state.currentBook;
      await loadCharList();
    }

    select.addEventListener('change', async () => {
      state.currentBook = select.value;
      state.currentChar = null;
      await loadCharList();
    });
  }

  async function loadCharList() {
    const list = qs('#char-list');
    clear(list);
    if (!state.currentBook) return;

    const data = await getJSON('/paddle/char_list', { book: state.currentBook });
    if (!data.success) return;
    state.charItems = data.chars || [];
    renderCharList();

    if (state.presetChar && state.charItems.some(item => item.char === state.presetChar)) {
      state.currentChar = state.presetChar;
      state.presetChar = null;
      renderCharList();
      await loadItems();
    }
  }

  function renderCharList() {
    const list = qs('#char-list');
    const filter = (qs('#char-filter').value || '').trim();
    clear(list);

    const fragment = document.createDocumentFragment();
    for (const item of state.charItems) {
      if (filter && !item.char.includes(filter)) continue;
      const row = el('div', {
        className: `char-item${item.char === state.currentChar ? ' active' : ''}`,
        onclick: async () => {
          state.currentChar = item.char;
          renderCharList();
          await loadItems();
        },
      }, [
        el('span', { text: item.char }),
        el('span', { text: item.total }),
      ]);
      fragment.appendChild(row);
    }
    list.appendChild(fragment);
  }

  async function loadItems() {
    const grid = qs('#grid');
    const title = qs('#char-title');
    const stat = qs('#char-stat');
    clear(grid);

    if (!state.currentBook || !state.currentChar) {
      text(title, '请选择一个字');
      text(stat, '');
      return;
    }

    const data = await getJSON('/paddle/items', {
      book: state.currentBook,
      char: state.currentChar,
    });
    if (!data.success) {
      grid.appendChild(el('div', { text: '加载失败' }));
      return;
    }

    text(title, `字：${state.currentChar}`);
    text(stat, `候选 ${data.items.length} 个`);

    const fragment = document.createDocumentFragment();
    for (const item of data.items) {
      fragment.appendChild(renderCard(item));
    }
    grid.appendChild(fragment);
  }

  function renderCard(item) {
    const card = el('div', { className: `card ${item.decision || 'pending'}` });
    const image = el('img', {
      className: 'thumb',
      src: item.image || item.segmented_path || '',
      alt: item.instance_id || '',
      loading: 'lazy',
    });
    const confidence = item.paddle_conf != null ? Number(item.paddle_conf).toFixed(3) : '0.000';
    const meta = el('div', { className: 'meta' }, [
      el('span', { text: `宽 ${item.width}` }),
      el('span', { text: `置信 ${confidence}` }),
    ]);
    const actions = el('div', { className: 'actions' }, [
      el('button', {
        className: 'btn need',
        type: 'button',
        text: '保留',
        onclick: () => updateDecision(item.instance_id, 'need', card),
      }),
      el('button', {
        className: 'btn drop',
        type: 'button',
        text: '不需要',
        onclick: () => updateDecision(item.instance_id, 'drop', card),
      }),
    ]);
    card.append(image, meta, actions);
    return card;
  }

  async function updateDecision(instanceId, decision, card) {
    const data = await postJSON('/paddle/decision', {
      book: state.currentBook,
      char: state.currentChar,
      instance_id: instanceId,
      decision,
    });
    if (!data.success) {
      alert(`更新失败: ${data.error || '未知错误'}`);
      return;
    }
    card.classList.remove('need', 'drop');
    card.classList.add(decision);
  }

  qs('#char-filter').addEventListener('input', renderCharList);
  loadBooks().catch(error => {
    console.error('Paddle 页面初始化失败:', error);
    text(qs('#char-title'), `加载失败: ${error.message}`);
  });
})();
