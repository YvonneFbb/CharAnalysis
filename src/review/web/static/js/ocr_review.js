(function () {
  const API = window.ReviewApi;
  const PAGE_SIZE = 20;

  let currentBook = null;
  let currentChar = null;
  let currentPage = 1;
  let hasMore = false;
  let totalCandidates = 0;
  let sortByWidth = true;
  let showMismatch = false;
  let deepLinkChar = null;
  let parentDirtyNotified = false;

  function qs(id) {
    return document.getElementById(id);
  }

  function bindControls() {
    qs('book-select').addEventListener('change', event => {
      loadBook(event.target.value);
    });
    qs('close-review-modal-btn').addEventListener('click', closeModal);
    qs('prev-page').addEventListener('click', () => loadPage(currentPage - 1));
    qs('next-page').addEventListener('click', () => loadPage(currentPage + 1));
    qs('sort-width-toggle').addEventListener('change', event => {
      sortByWidth = !!event.target.checked;
      if (currentChar) loadPage(1);
    });
    qs('show-mismatch-toggle').addEventListener('change', event => {
      showMismatch = !!event.target.checked;
      if (currentChar) loadPage(1);
    });

    document.addEventListener('keydown', event => {
      const modal = qs('review-modal');
      if (!modal.classList.contains('active')) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        closeModal();
      } else if (event.key === 'ArrowLeft' && currentPage > 1) {
        event.preventDefault();
        loadPage(currentPage - 1);
      } else if (event.key === 'ArrowRight' && hasMore) {
        event.preventDefault();
        loadPage(currentPage + 1);
      }
    });
  }

  async function initialize() {
    bindControls();
    await loadBooks();
    await handleDeepLink();
  }

  async function loadBooks() {
    try {
      const data = await API.getJSON('/filter/books');
      const select = qs('book-select');
      select.innerHTML = '<option value="">-- 请选择一本书 --</option>';
      (data.books || []).forEach(book => {
        const option = document.createElement('option');
        option.value = book.name;
        option.textContent = `${book.name} (${book.total_chars}字, ${book.total_instances}实例)`;
        select.appendChild(option);
      });
    } catch (error) {
      console.error('加载书籍失败:', error);
      alert('无法连接到后端服务');
    }
  }

  async function handleDeepLink() {
    const params = new URLSearchParams(window.location.search);
    const book = params.get('book');
    deepLinkChar = params.get('char');
    if (!book) return;
    qs('book-select').value = book;
    await loadBook(book);
    if (deepLinkChar) {
      openFilterModal(deepLinkChar);
    }
  }

  function setBookStats(data) {
    qs('stat-chars').textContent = data.total_chars || 0;
    qs('stat-instances').textContent = data.total_instances || 0;
    qs('stat-processed').textContent = data.processed_chars || 0;
    qs('stat-accepted').textContent = data.accepted_instances || 0;
    qs('stat-rejected').textContent = data.rejected_instances || 0;
    qs('book-info').classList.add('active');
  }

  async function loadBook(bookName) {
    currentBook = bookName || null;
    currentChar = null;
    currentPage = 1;
    totalCandidates = 0;
    hasMore = false;

    if (!currentBook) {
      qs('book-info').classList.remove('active');
      qs('methods-container').classList.remove('active');
      qs('methods-container').innerHTML = '<div class="loading">请选择书籍</div>';
      return;
    }

    qs('methods-container').classList.add('active');
    qs('methods-container').innerHTML = '<div class="loading">加载中...</div>';
    try {
      const data = await API.getJSON(`/filter/book/${encodeURIComponent(currentBook)}`);
      setBookStats(data);
      renderMethods(data.methods || []);
    } catch (error) {
      console.error('加载书籍详情失败:', error);
      qs('methods-container').innerHTML = '<div class="loading">加载失败</div>';
    }
  }

  async function refreshCurrentBookSidebar() {
    if (!currentBook) return;
    const container = qs('methods-container');
    const scrollTop = container ? container.scrollTop : 0;
    try {
      const data = await API.getJSON(`/filter/book/${encodeURIComponent(currentBook)}`);
      setBookStats(data);
      renderMethods(data.methods || []);
      if (container) {
        container.scrollTop = scrollTop;
      }
    } catch (error) {
      console.error('刷新侧栏失败:', error);
    }
  }

  function renderMethods(methods) {
    const container = qs('methods-container');
    container.innerHTML = '';

    methods.forEach(method => {
      const section = document.createElement('div');
      section.className = 'method-section expanded';

      const header = document.createElement('div');
      header.className = 'method-header';
      header.addEventListener('click', () => section.classList.toggle('expanded'));

      const headerInfo = document.createElement('div');
      const title = document.createElement('span');
      title.className = 'method-title';
      title.textContent = `${method.id}. ${method.name}`;
      const description = document.createElement('div');
      description.className = 'method-meta';
      description.textContent = method.description || '';
      headerInfo.appendChild(title);
      headerInfo.appendChild(description);

      const count = document.createElement('div');
      count.className = 'method-meta';
      count.textContent = `${(method.chars || []).length} 个字符`;

      header.appendChild(headerInfo);
      header.appendChild(count);

      const charsContainer = document.createElement('div');
      charsContainer.className = 'method-chars';

      (method.chars || []).forEach(charInfo => {
        const item = document.createElement('div');
        item.className = 'char-item';
        if ((charInfo.accepted || 0) > 0) {
          item.classList.add('reviewed');
        } else if ((charInfo.processed || 0) > 0) {
          item.classList.add('partially-reviewed');
        }
        item.dataset.char = charInfo.char;
        item.addEventListener('click', () => openFilterModal(charInfo.char));

        const charText = document.createElement('div');
        charText.className = 'char-text';
        charText.textContent = charInfo.char;

        const countText = document.createElement('div');
        countText.className = 'char-count';
        countText.textContent = `${charInfo.matched || 0} 匹配 / ${charInfo.prepared || 0} 已准备 / ${charInfo.accepted || 0} 已接受 / ${charInfo.count || 0} OCR`;

        item.appendChild(charText);
        item.appendChild(countText);
        charsContainer.appendChild(item);
      });

      section.appendChild(header);
      section.appendChild(charsContainer);
      container.appendChild(section);
    });

    if (!methods.length) {
      container.innerHTML = '<div class="loading">没有可显示的字符</div>';
    }
  }

  function openFilterModal(char) {
    currentChar = char;
    currentPage = 1;
    totalCandidates = 0;
    qs('modal-title').textContent = `筛选字符：${char}`;
    qs('modal-summary').textContent = '';
    qs('review-modal').classList.add('active');
    loadPage(1);
  }

  async function loadPage(page) {
    if (!currentBook || !currentChar || page < 1) return;
    currentPage = page;
    qs('modal-loading').style.display = 'block';
    qs('instances-grid').classList.add('is-hidden');

    try {
      const data = await API.getJSON('/filter/items', {
        book: currentBook,
        char: currentChar,
        page,
        page_size: PAGE_SIZE,
        sort: sortByWidth ? 'width_desc' : 'default',
        include_mismatch: showMismatch ? 1 : 0,
      });
      hasMore = !!data.has_more;
      totalCandidates = data.total_candidates || 0;
      qs('modal-title').textContent = `筛选字符：${currentChar}（OCR ${totalCandidates} 项）`;
      qs('page-info').textContent = `第 ${currentPage} 页`;
      qs('prev-page').disabled = currentPage <= 1;
      qs('next-page').disabled = !hasMore;
      qs('modal-summary').textContent = `本页 ${data.items.length} 项`;
      renderInstances(data.items || []);
    } catch (error) {
      console.error('加载筛选候选失败:', error);
      qs('instances-grid').innerHTML = '<div class="empty-card">加载失败</div>';
      qs('instances-grid').classList.remove('is-hidden');
      qs('prev-page').disabled = currentPage <= 1;
      qs('next-page').disabled = true;
    } finally {
      qs('modal-loading').style.display = 'none';
    }
  }

  function createStatusBadge(item) {
    const badge = document.createElement('div');
    badge.className = 'instance-badges';

    const filterTag = document.createElement('span');
    const status = item.filter_status || 'pending';
    filterTag.className = `instance-tag instance-tag-${status}`;
    filterTag.textContent =
      status === 'accepted' ? '已接受' :
      status === 'rejected' ? '已拒绝' : '待定';
    badge.appendChild(filterTag);

    const matchTag = document.createElement('span');
    const match = item.reocr_matches;
    matchTag.className = `instance-tag ${
      item.reocr_state === 'error'
        ? 'instance-tag-error'
        : (match === true ? 'instance-tag-match' : match === false ? 'instance-tag-mismatch' : 'instance-tag-unknown')
    }`;
    matchTag.textContent =
      item.reocr_state === 'error' ? 'reOCR 失败' :
      match === true ? 'reOCR 匹配' :
      match === false ? 'reOCR 不匹配' : 'reOCR 准备中';
    badge.appendChild(matchTag);

    return badge;
  }

  function renderInstances(items) {
    const grid = qs('instances-grid');
    grid.innerHTML = '';
    if (!items.length) {
      grid.innerHTML = '<div class="empty-card">当前条件下没有候选</div>';
      grid.classList.remove('is-hidden');
      return;
    }

    items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'instance-item';
      applyCardState(card, item);

      const previewWrap = document.createElement('div');
      previewWrap.className = 'instance-preview';
      if (item.preview_image) {
        const img = document.createElement('img');
        img.src = item.preview_image;
        img.alt = item.char || currentChar;
        previewWrap.appendChild(img);
      } else {
        previewWrap.innerHTML = '<div class="preview-missing">预览缺失</div>';
      }

      const info = document.createElement('div');
      info.className = 'instance-info';
      const sizeText = item.segmented_width && item.segmented_width !== item.width
        ? `OCR宽 ${item.width || 0}px · 高 ${item.height || 0}px · Segment宽 ${item.segmented_width || 0}px`
        : `OCR宽 ${item.width || 0}px · 高 ${item.height || 0}px`;
      info.innerHTML = [
        `册${String(item.volume || '').padStart(2, '0')} · ${item.page || '-'}`,
        sizeText,
        item.reocr_text ? `reOCR: ${item.reocr_text} (${formatConfidence(item.reocr_confidence)})` : 'reOCR: -',
      ].map(text => `<div>${text}</div>`).join('');

      const badges = createStatusBadge(item);

      const actions = document.createElement('div');
      actions.className = 'instance-actions';
      actions.appendChild(createDecisionButton('接受', 'accepted', item, card));
      actions.appendChild(createDecisionButton('拒绝', 'rejected', item, card));
      if ((item.filter_status || 'pending') !== 'pending') {
        actions.appendChild(createDecisionButton('待定', 'pending', item, card));
      }

      if (item.error) {
        const error = document.createElement('div');
        error.className = 'instance-error';
        error.textContent = item.error;
        card.appendChild(error);
      }

      card.appendChild(previewWrap);
      card.appendChild(info);
      card.appendChild(badges);
      card.appendChild(actions);
      grid.appendChild(card);
    });

    grid.classList.remove('is-hidden');
  }

  function createDecisionButton(label, status, item, card) {
    const button = document.createElement('button');
    button.className = 'instance-action-btn';
    if ((item.filter_status || 'pending') === status) {
      button.classList.add('active');
    }
    button.textContent = label;
    button.addEventListener('click', async event => {
      event.stopPropagation();
      await updateDecision(item, status, card);
    });
    return button;
  }

  function applyCardState(card, item) {
    card.classList.remove('status-pending', 'status-accepted', 'status-rejected');
    card.classList.add(`status-${item.filter_status || 'pending'}`);
  }

  async function updateDecision(item, status, card) {
    const buttons = Array.from(card.querySelectorAll('.instance-action-btn'));
    buttons.forEach(button => { button.disabled = true; });
    try {
      const data = await API.postJSON('/filter/decision', {
        book: currentBook,
        char: currentChar,
        instance_id: item.instance_id,
        status,
      });
      Object.assign(item, data.item || {});
      card.innerHTML = '';
      applyCardState(card, item);

      const previewWrap = document.createElement('div');
      previewWrap.className = 'instance-preview';
      if (item.preview_image) {
        const img = document.createElement('img');
        img.src = item.preview_image;
        img.alt = item.char || currentChar;
        previewWrap.appendChild(img);
      } else {
        previewWrap.innerHTML = '<div class="preview-missing">预览缺失</div>';
      }

      const info = document.createElement('div');
      info.className = 'instance-info';
      info.innerHTML = [
        `册${String(item.volume || '').padStart(2, '0')} · ${item.page || '-'}`,
        `宽 ${item.width || 0}px · 高 ${item.height || 0}px`,
        item.reocr_text ? `reOCR: ${item.reocr_text} (${formatConfidence(item.reocr_confidence)})` : 'reOCR: -',
      ].map(text => `<div>${text}</div>`).join('');

      card.appendChild(previewWrap);
      card.appendChild(info);
      card.appendChild(createStatusBadge(item));

      const actions = document.createElement('div');
      actions.className = 'instance-actions';
      actions.appendChild(createDecisionButton('接受', 'accepted', item, card));
      actions.appendChild(createDecisionButton('拒绝', 'rejected', item, card));
      if ((item.filter_status || 'pending') !== 'pending') {
        actions.appendChild(createDecisionButton('待定', 'pending', item, card));
      }
      card.appendChild(actions);
      notifyParentDirty();
    } catch (error) {
      console.error('更新筛选状态失败:', error);
      alert(`更新失败：${error.message}`);
    } finally {
      refreshCurrentBookSidebar();
    }
  }

  function closeModal() {
    qs('review-modal').classList.remove('active');
    qs('instances-grid').classList.add('is-hidden');
    currentChar = null;
  }

  function notifyParentDirty() {
    if (parentDirtyNotified) return;
    if (window.top === window) return;
    window.top.postMessage({ type: 'filter-state-dirty' }, window.location.origin);
    parentDirtyNotified = true;
  }

  function formatConfidence(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
    return Number(value).toFixed(3);
  }

  document.addEventListener('DOMContentLoaded', initialize);
})();
