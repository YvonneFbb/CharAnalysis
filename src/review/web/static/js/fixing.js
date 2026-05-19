(function () {
  const { getJSON, url } = window.ReviewApi;
  const { qs, clear, text, el, option } = window.ReviewDom;
  const TILE = 96;

  const state = {
    currentBook: null,
    filterModalDirty: false,
    segmentModalDirty: false,
  };

  async function loadBooks() {
    const select = qs('#book-select');
    clear(select);
    const data = await getJSON('/books_simple');
    if (!data.success) return;

    const fragment = document.createDocumentFragment();
    for (const book of data.books || []) {
      fragment.appendChild(option(book.name, book.name));
    }
    select.appendChild(fragment);

    if ((data.books || []).length > 0) {
      state.currentBook = data.books[0].name;
      select.value = state.currentBook;
    }
    select.addEventListener('change', () => {
      state.currentBook = select.value;
      reloadFixing();
    });
  }

  async function reloadFixing() {
    const grid = qs('#grid');
    const threshold = Math.max(0, Math.min(255, parseInt(qs('#bw-threshold').value || '128', 10)));
    if (!state.currentBook) return;

    try {
      const data = await getJSON('/fixing_items', {
        book: state.currentBook,
        all: 1,
        image: 'thumb',
        threshold,
        use_fixed_box: 1,
        tile: TILE,
      });
      if (!data.success) {
        clear(grid);
        grid.appendChild(el('div', { text: '加载失败' }));
        return;
      }

      text(qs('#summary'), `共 ${data.total} 项`);
      const Lb = data.fixed_box && data.fixed_box.L_b ? data.fixed_box.L_b : 0;
      text(qs('#fixed-box-stat'), Lb ? `固定框 L_b=${Lb}` : '');
      refreshMontage();
      renderItems(data.items || []);
      renderMissing(data.missing_chars || []);
    } catch (error) {
      console.error('Fixing 数据加载失败:', error);
      clear(grid);
      grid.appendChild(el('div', { text: `加载失败: ${error.message}` }));
    }
  }

  function renderItems(items) {
    const grid = qs('#grid');
    clear(grid);
    const fragment = document.createDocumentFragment();
    for (const item of items) {
      fragment.appendChild(renderCard(item));
    }
    grid.appendChild(fragment);
  }

  function renderCard(item) {
    const decision = item.decision || 'unknown';
    const status = item.status || 'unreviewed';
    const cardClass = decision === 'drop'
      ? 'dropped'
      : (status === 'pending_manual' ? 'pending' : (status === 'confirmed' ? 'confirmed' : 'unreviewed'));

    const card = el('div', {
      className: `card ${cardClass}`,
      onclick: () => openSegmentModalForItem(item),
    });
    card.appendChild(renderStatusBadge(decision, status));

    if (item.thumb) {
      card.appendChild(el('img', {
        className: 'thumb',
        src: item.thumb,
        alt: `${item.char}-${item.instance_id}`,
        loading: 'lazy',
      }));
    }
    card.appendChild(renderCardLabel(item));
    return card;
  }

  function renderStatusBadge(decision, status) {
    let statusText = '未处理';
    let statusClass = 'status-unreviewed';
    if (decision === 'drop') {
      statusText = '不需要';
      statusClass = 'status-drop';
    } else if (status === 'confirmed') {
      statusText = '已确认';
      statusClass = 'status-confirmed';
    } else if (status === 'pending_manual') {
      statusText = '待手动';
      statusClass = 'status-pending';
    }
    return el('div', { className: `status-badge ${statusClass}`, text: statusText });
  }

  function renderCardLabel(item) {
    const charName = el('span', {
      className: 'char-name',
      title: '打开 Review 微调',
      text: item.char,
      onclick: event => {
        event.stopPropagation();
        openSegmentModalForItem(item);
      },
    });
    const reviewButton = el('button', {
      type: 'button',
      text: 'Review',
      onclick: event => {
        event.stopPropagation();
        openSegmentModalForItem(item);
      },
    });
    const ocrButton = el('button', {
      type: 'button',
      text: 'Filter',
      onclick: event => {
        event.stopPropagation();
        openFilterModalForChar(item.char);
      },
    });
    return el('div', { className: 'thumb-label' }, [
      charName,
      el('div', { className: 'thumb-actions' }, [reviewButton, ocrButton]),
    ]);
  }

  function renderMissing(missingChars) {
    const grid = qs('#missing-grid');
    clear(grid);
    text(qs('#missing-summary'), `共 ${missingChars.length} 个`);

    if (missingChars.length === 0) {
      const empty = el('div', { text: '当前书籍没有缺失字' });
      empty.style.fontSize = '12px';
      empty.style.color = '#6b737c';
      grid.appendChild(empty);
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const char of missingChars) {
      const button = el('button', {
        className: 'missing-chip',
        type: 'button',
        text: char,
        onclick: () => openFilterModalForChar(char),
      });
      fragment.appendChild(el('div', { className: 'missing-item' }, [button]));
    }
    grid.appendChild(fragment);
  }

  function requireBook() {
    if (state.currentBook) return true;
    alert('请先选择书籍');
    return false;
  }

  function openFilterModal(targetChar) {
    if (!requireBook()) return;
    state.filterModalDirty = false;
    const params = new URLSearchParams({ book: state.currentBook });
    if (targetChar) params.set('char', targetChar);
    if (new URLSearchParams(window.location.search).has('debug')) {
      params.set('debug', '1');
    }
    qs('#ocr-frame').src = `/filter?${params.toString()}`;
    qs('#ocr-modal').classList.add('active');
  }

  function openFilterModalForChar(char) {
    openFilterModal(char);
  }

  function closeFilterModal() {
    const frame = qs('#ocr-frame');
    frame.src = 'about:blank';
    qs('#ocr-modal').classList.remove('active');
    const shouldRefresh = state.filterModalDirty;
    state.filterModalDirty = false;
    if (shouldRefresh) {
      setTimeout(reloadFixing, 200);
    }
  }

  function refreshFilterFrame() {
    const frame = qs('#ocr-frame');
    if (frame.src) frame.src = frame.src;
  }

  function openSegmentModalForItem(item) {
    if (!requireBook()) return;
    state.segmentModalDirty = false;
    const params = new URLSearchParams({ book: state.currentBook });
    if (item && item.char) params.set('char', item.char);
    if (item && item.instance_id) params.set('instance_id', item.instance_id);
    if (new URLSearchParams(window.location.search).has('debug')) {
      params.set('debug', '1');
    }
    qs('#segment-frame').src = `/segment_review?${params.toString()}`;
    qs('#segment-modal').classList.add('active');
  }

  function closeSegmentModal() {
    const frame = qs('#segment-frame');
    frame.src = 'about:blank';
    qs('#segment-modal').classList.remove('active');
    const shouldRefresh = state.segmentModalDirty;
    state.segmentModalDirty = false;
    if (shouldRefresh) {
      setTimeout(reloadFixing, 200);
    }
  }

  function refreshSegmentFrame() {
    const frame = qs('#segment-frame');
    if (frame.src) frame.src = frame.src;
  }

  function refreshMontage() {
    if (!state.currentBook) return;
    const montageUrl = url('/fixing_montage', {
      book: state.currentBook,
      use_fixed_box: 1,
      tile: TILE,
      cols: 50,
      ts: Date.now(),
    });
    const image = qs('#montage-img');
    image.src = montageUrl;
    image.onclick = () => openMontageModal(montageUrl);
  }

  function openMontageModal(src) {
    qs('#montage-modal-img').src = src;
    qs('#montage-modal').classList.add('active');
  }

  function closeMontageModal() {
    qs('#montage-modal-img').src = '';
    qs('#montage-modal').classList.remove('active');
  }

  function bindStaticControls() {
    qs('#reload-fixing-btn').addEventListener('click', reloadFixing);
    qs('#open-ocr-modal-btn').addEventListener('click', () => openFilterModal());
    qs('#refresh-montage-btn').addEventListener('click', refreshMontage);
    qs('#refresh-ocr-frame-btn').addEventListener('click', refreshFilterFrame);
    qs('#close-ocr-modal-btn').addEventListener('click', closeFilterModal);
    qs('#refresh-segment-frame-btn').addEventListener('click', refreshSegmentFrame);
    qs('#close-segment-modal-btn').addEventListener('click', closeSegmentModal);
    qs('#close-montage-modal-btn').addEventListener('click', closeMontageModal);
  }

  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    if (qs('#montage-modal').classList.contains('active')) {
      closeMontageModal();
      return;
    }
    if (qs('#segment-modal').classList.contains('active')) {
      closeSegmentModal();
      return;
    }
    if (qs('#ocr-modal').classList.contains('active')) {
      closeFilterModal();
    }
  });

  window.addEventListener('message', event => {
    if (event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.type === 'filter-state-dirty') {
      state.filterModalDirty = true;
    } else if (data.type === 'close-ocr-modal') {
      closeFilterModal();
    } else if (data.type === 'segment-state-dirty') {
      state.segmentModalDirty = true;
    } else if (data.type === 'close-segment-modal') {
      closeSegmentModal();
    }
  });

  (async function init() {
    bindStaticControls();
    await loadBooks();
    await reloadFixing();
  })().catch(error => {
    console.error('Fixing 页面初始化失败:', error);
    text(qs('#summary'), `加载失败: ${error.message}`);
  });
})();
