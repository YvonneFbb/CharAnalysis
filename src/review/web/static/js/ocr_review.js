(function () {
const DEBUG = new URLSearchParams(window.location.search).has('debug');
        const API_BASES = window.ReviewApi.apiBases({ includeLocalhost: true });
        const API_BASE = API_BASES[0];
        let currentBook = null;
        let currentChar = null;
        let currentPage = 1;
        let totalPages = 1;
        let instancesData = [];
        let reviewResults = { version: 2, books: {} };
        let sortByWidth = false;
        let sortByPaddle = false;
        let saveTimer = null;
        const SAVE_DEBOUNCE_MS = 1500;
        const dirtyBooks = new Set(); // 记录有改动的书籍
        const dirtyCharsByBook = new Map(); // 记录每本书改动的字符
        let syncStatusEl = null;

        function setSyncStatus(text, isError = false) {
            if (!syncStatusEl) {
                syncStatusEl = document.createElement('div');
                syncStatusEl.style.cssText = [
                    'position: fixed',
                    'top: 10px',
                    'right: 12px',
                    'z-index: 9999',
                    'padding: 6px 10px',
                    'border-radius: 6px',
                    'font-size: 12px',
                    'background: rgba(240, 248, 255, 0.95)',
                    'color: #1b4d8c',
                    'box-shadow: 0 2px 6px rgba(0,0,0,0.1)'
                ].join(';');
                document.body.appendChild(syncStatusEl);
            }
            syncStatusEl.textContent = text;
            syncStatusEl.style.background = isError ? 'rgba(255, 238, 238, 0.95)' : 'rgba(240, 248, 255, 0.95)';
            syncStatusEl.style.color = isError ? '#8b1d1d' : '#1b4d8c';
        }

        function markDirtyChar(bookName, char) {
            if (!bookName || !char) return;
            dirtyBooks.add(bookName);
            if (!dirtyCharsByBook.has(bookName)) {
                dirtyCharsByBook.set(bookName, new Set());
            }
            dirtyCharsByBook.get(bookName).add(char);
        }

        function clearDirtyState() {
            dirtyBooks.clear();
            dirtyCharsByBook.clear();
        }

        const store = window.OcrReviewStore.createStore({
            apiBase: API_BASE,
            apiBases: API_BASES,
            debug: DEBUG,
            getReviewResults: () => reviewResults,
            setReviewResults: value => { reviewResults = value; },
            getDirtyBooks: () => dirtyBooks,
            getDirtyCharsByBook: () => dirtyCharsByBook,
            clearDirtyState,
            setSyncStatus,
            persistReviewResults,
        });

        function bindStaticControls() {
            document.getElementById('export-results-btn').addEventListener('click', exportResults);
            document.getElementById('import-results-btn').addEventListener('click', importResults);
            document.getElementById('import-file').addEventListener('change', handleImport);
            document.getElementById('close-review-modal-btn').addEventListener('click', closeModal);
            document.getElementById('prev-page').addEventListener('click', () => loadPage(currentPage - 1));
            document.getElementById('next-page').addEventListener('click', () => loadPage(currentPage + 1));
            document.getElementById('sort-width-toggle').addEventListener('change', event => toggleWidthSort(event.target.checked));
            document.getElementById('sort-paddle-toggle').addEventListener('change', event => togglePaddleSort(event.target.checked));
            document.getElementById('clear-selections-btn').addEventListener('click', clearAllSelections);
        }

        // 初始化
        document.addEventListener('DOMContentLoaded', async function() {
            bindStaticControls();
            await loadBooks();
            store.initializeResults();
            if (DEBUG) {
                console.log('[OCR] API_BASES:', API_BASES);
                try {
                    const pingResp = await fetch(`${API_BASE}/ping`);
                    const pingData = await pingResp.json();
                    console.log('[OCR] ping:', pingData);
                } catch (e) {
                    console.warn('[OCR] ping failed:', e);
                }
            }
            await handleDeepLink();

            // 定期自动保存到服务器（每 30 秒）
            setInterval(() => {
                saveToServer();
            }, 30000);

            // 页面关闭前保存到服务器
            window.addEventListener('beforeunload', function(e) {
                const payload = store.buildDirtyPayload();
                if (!payload) return;
                store.sendBeaconPayload(payload);
            });

            // 添加键盘快捷键
            document.addEventListener('keydown', function(e) {
                const modal = document.getElementById('review-modal');
                const isModalOpen = modal.classList.contains('active');

                // ESC：优先关闭当前模态；如果未打开模态则通知父级关闭 iframe 弹窗
                if (e.key === 'Escape') {
                    if (isModalOpen) {
                        e.preventDefault();
                        closeModal();
                        return;
                    }
                    if (window.self !== window.top) {
                        e.preventDefault();
                        try {
                            window.top.postMessage({ type: 'close-ocr-modal' }, window.location.origin);
                        } catch (_) {}
                    }
                    return;
                }

                if (!isModalOpen) return;

                // 左方向键：上一页
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    if (currentPage > 1) {
                        loadPage(currentPage - 1);
                    }
                }

                // 右方向键：下一页
                if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    if (currentPage < totalPages) {
                        loadPage(currentPage + 1);
                    }
                }
            });
        });

        // 加载书籍列表
        async function loadBooks() {
            try {
                const response = await fetch(`${API_BASE}/books`);
                const data = await response.json();

                const select = document.getElementById('book-select');
                select.innerHTML = '<option value="">-- 请选择一本书 --</option>';

                data.books.forEach(book => {
                    const option = document.createElement('option');
                    option.value = book.name;
                    option.textContent = `${book.name} (${book.total_chars}字, ${book.total_instances}实例)`;
                    select.appendChild(option);
                });

                select.addEventListener('change', function() {
                    loadBook(this.value);
                });
            } catch (error) {
                console.error('加载书籍列表失败:', error);
                alert('无法连接到服务器，请确保后端服务已启动');
            }
        }

        // 加载书籍字符
        async function loadBook(bookName) {
            if (!bookName) {
                document.getElementById('book-info').classList.remove('active');
                document.getElementById('methods-container').classList.remove('active');
                return;
            }

            // 关键修复：切换书籍前，先上传当前书籍的数据
            if (currentBook && currentBook !== bookName) {
                console.log(`切换书籍：从 "${currentBook}" 到 "${bookName}"，先上传当前数据...`);
                await saveToServer();
            }

            currentBook = bookName;

            // 从服务器同步新书籍的最新数据
            await syncBookFromServer(bookName);

            // 刷新显示
            await refreshBookDisplay();
        }

        // 刷新书籍显示（不从服务器同步）
        async function refreshBookDisplay() {
            if (!currentBook) return;

            try {
                const response = await fetch(`${API_BASE}/book/${encodeURIComponent(currentBook)}`);
                const data = await response.json();

                // 更新统计信息
                document.getElementById('stat-chars').textContent = data.total_chars;
                document.getElementById('stat-instances').textContent = data.total_instances;
                document.getElementById('book-info').classList.add('active');

                // 更新审查进度
                updateReviewProgress();

                // 渲染分类方法
                renderMethods(data.methods);
                document.getElementById('methods-container').classList.add('active');
            } catch (error) {
                console.error('刷新书籍显示失败:', error);
            }
        }

        async function handleDeepLink() {
            try {
                const params = new URLSearchParams(window.location.search);
                const deepBook = params.get('book');
                const deepChar = params.get('char');
                if (!deepBook) return;
                const select = document.getElementById('book-select');
                if (select) {
                    const option = Array.from(select.options).find(opt => opt.value === deepBook);
                    if (option) select.value = deepBook;
                }
                await loadBook(deepBook);
                if (deepChar) {
                    const charItem = document.querySelector(`.char-item[data-char="${deepChar}"]`);
                    if (charItem) {
                        charItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        const count = parseInt(charItem.dataset.count || '0', 10);
                        openReviewModal(deepChar, count || 0);
                    }
                }
            } catch (error) {
                console.warn('深链处理失败:', error);
            }
        }

        // 渲染分类方法
        function renderMethods(methods) {
            // 保存当前的展开状态
            const container = document.getElementById('methods-container');
            const expandedMethods = new Set();

            if (container.querySelectorAll('.method-section').length > 0) {
                container.querySelectorAll('.method-section.expanded').forEach(section => {
                    const methodTitle = section.querySelector('.method-title').textContent;
                    expandedMethods.add(methodTitle);
                });
            }

            container.innerHTML = '';

            methods.forEach(method => {
                const section = document.createElement('div');
                section.className = 'method-section';

                // 检查这个方法是否之前是展开的
                const methodTitle = `${method.id}. ${method.name}`;
                if (expandedMethods.has(methodTitle)) {
                    section.classList.add('expanded');
                }

                const header = document.createElement('div');
                header.className = 'method-header';
                header.addEventListener('click', () => toggleMethod(header));

                const headerInfo = document.createElement('div');
                const title = document.createElement('span');
                title.className = 'method-title';
                title.textContent = methodTitle;
                const description = document.createElement('div');
                description.className = 'method-meta';
                description.textContent = method.description || '';
                headerInfo.appendChild(title);
                headerInfo.appendChild(description);

                const count = document.createElement('div');
                count.className = 'method-meta';
                count.textContent = `${method.chars.length} 个字符`;

                header.appendChild(headerInfo);
                header.appendChild(count);

                const charsContainer = document.createElement('div');
                charsContainer.className = 'method-chars';

                method.chars.forEach(charInfo => {
                    const reviewed = isCharReviewed(charInfo.char);
                    const item = document.createElement('div');
                    item.className = `char-item${reviewed ? ' reviewed' : ''}`;
                    item.dataset.char = charInfo.char;
                    item.dataset.count = charInfo.count;
                    item.addEventListener('click', () => openReviewModal(charInfo.char, charInfo.count));

                    const charText = document.createElement('div');
                    charText.className = 'char-text';
                    charText.textContent = charInfo.char;

                    const charCount = document.createElement('div');
                    charCount.className = 'char-count';
                    charCount.textContent = `${charInfo.count} 个`;

                    item.appendChild(charText);
                    item.appendChild(charCount);

                    if (reviewed) {
                        const badge = document.createElement('div');
                        badge.className = 'reviewed-badge';
                        badge.textContent = '✓ 已审查';
                        item.appendChild(badge);
                    }

                    charsContainer.appendChild(item);
                });

                section.appendChild(header);
                section.appendChild(charsContainer);
                container.appendChild(section);
            });

            // 只在第一次加载时展开所有（没有任何历史状态）
            if (expandedMethods.size === 0 && methods.length > 0) {
                container.querySelectorAll('.method-section').forEach(section => {
                    section.classList.add('expanded');
                });
            }
        }

        // 切换分类方法展开/折叠
        function toggleMethod(header) {
            const section = header.parentElement;
            section.classList.toggle('expanded');
        }

        // 打开审查模态窗口
        function openReviewModal(char, totalCount) {
            currentChar = char;
            currentPage = 1;

            document.getElementById('modal-title').textContent = `审查字符：${char} (${totalCount} 个实例)`;
            document.getElementById('review-modal').classList.add('active');

            loadPage(1);
        }

        // 加载指定页的实例
        async function loadPage(page) {
            if (page < 1 || (totalPages > 0 && page > totalPages)) return;

            // 切换页面前，自动保存当前页的审查结果
            if (currentChar && document.querySelectorAll('.instance-item').length > 0) {
                saveReview();
            }

            currentPage = page;
            document.getElementById('modal-loading').style.display = 'block';
            document.getElementById('instances-grid').style.display = 'none';

            try {
                const response = await fetch(`${API_BASE}/instances?book=${encodeURIComponent(currentBook)}&char=${encodeURIComponent(currentChar)}&page=${page}&page_size=20&sort=${getSortMode()}`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const data = await response.json();

                instancesData = data.instances;
                totalPages = data.total_pages;

                // 更新分页信息
                document.getElementById('page-info').textContent = `第 ${currentPage}/${totalPages} 页`;
                document.getElementById('prev-page').disabled = currentPage === 1;
                document.getElementById('next-page').disabled = currentPage === totalPages;

                // 渲染实例
                await renderInstances(data.instances);

                document.getElementById('modal-loading').style.display = 'none';
                document.getElementById('instances-grid').style.display = 'grid';
            } catch (error) {
                console.error('加载实例失败:', error);
                alert('加载实例失败，请重试');
                // 恢复界面可操作，避免遮罩挡住关闭按钮
                document.getElementById('modal-loading').style.display = 'none';
                document.getElementById('instances-grid').style.display = 'grid';
                document.getElementById('page-info').textContent = '加载失败';
            }
        }

        // 渲染实例
        async function renderInstances(instances) {
            const grid = document.getElementById('instances-grid');
            grid.innerHTML = '';

            for (const [localIndex, instance] of instances.entries()) {
                const instanceIndex = (instance && typeof instance.instance_index === 'number')
                    ? instance.instance_index
                    : ((currentPage - 1) * 20 + localIndex);
                
                const item = document.createElement('div');
                item.className = 'instance-item';
                item.dataset.arrayIndex = instanceIndex;  // 保存全局数组索引

                // 检查已保存的审查状态（使用v2格式）
                const charInstances = getCharInstances(currentBook, currentChar);
                if (charInstances[instanceIndex] === true) {
                    item.classList.add('selected');
                }

                item.innerHTML = `
                    <div class="instance-loading">加载中...</div>
                    <div class="instance-info">册${String(instance.volume).padStart(2, '0')} - ${instance.page}</div>
                    <div class="instance-status">${item.classList.contains('selected') ? '✓ 已选择' : '未选择'}</div>
                `;

                item.addEventListener('click', function() {
                    toggleInstance(this);
                });

                grid.appendChild(item);

                // 异步裁切图片
                cropImage(instanceIndex, item);
            }
        }

        function toggleWidthSort(checked) {
            if (sortByPaddle) {
                const toggle = document.getElementById('sort-width-toggle');
                if (toggle) toggle.checked = false;
                return;
            }
            sortByWidth = !!checked;
            loadPage(1);
        }

        function togglePaddleSort(checked) {
            sortByPaddle = !!checked;
            const widthToggle = document.getElementById('sort-width-toggle');
            if (sortByPaddle) {
                sortByWidth = false;
                if (widthToggle) {
                    widthToggle.checked = false;
                    widthToggle.disabled = true;
                }
            } else if (widthToggle) {
                widthToggle.disabled = false;
            }
            loadPage(1);
        }

        function getSortMode() {
            if (sortByPaddle) return 'paddle';
            if (sortByWidth) return 'width_desc';
            return 'default';
        }

        // 裁切图片
        async function cropImage(arrayIndex, itemElement) {
            try {
                const response = await fetch(`${API_BASE}/crop`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        index: arrayIndex,
                        padding: 5
                    })
                });

                const data = await response.json();

                if (data.success) {
                    const loadingDiv = itemElement.querySelector('.instance-loading');
                    const img = document.createElement('img');
                    img.src = data.image;
                    img.alt = currentChar;
                    loadingDiv.replaceWith(img);
                }
            } catch (error) {
                console.error('裁切图片失败:', error);
            }
        }

        // 持久化（当前只保留内存状态）
        function persistReviewResults() {
            // 服务器是唯一数据源，本地只保留内存状态
            return;
        }

        // 节流上传：每次修改后启动/重置定时器
        function scheduleSaveToServer() {
            if (saveTimer) {
                clearTimeout(saveTimer);
            }
            saveTimer = setTimeout(() => {
                saveToServer();
            }, SAVE_DEBOUNCE_MS);
        }

        // 切换实例状态（v2: 带时间戳更新）
        function toggleInstance(item) {
            if (item.classList.contains('selected')) {
                item.classList.remove('selected');
                item.querySelector('.instance-status').textContent = '未选择';
            } else {
                item.classList.add('selected');
                item.querySelector('.instance-status').textContent = '✓ 已选择';
            }

            // 更新内存状态（使用新的时间戳更新函数）
            const arrayIndex = item.dataset.arrayIndex;
            const selected = item.classList.contains('selected');

            updateCharInstance(currentBook, currentChar, arrayIndex, selected);
            persistReviewResults();
            markDirtyChar(currentBook, currentChar);
            // 立即触发上传，确保后端实时更新
            saveToServer();
        }

        // 取消全部选择（跨页）
        function clearAllSelections() {
            const items = document.querySelectorAll('.instance-item');
            if (!items.length) return;
            if (!currentBook || !currentChar) return;

            items.forEach(item => {
                item.classList.remove('selected');
                item.querySelector('.instance-status').textContent = '未选择';
            });

            // 清空该字符的所有选择（跨页）
            setCharData(currentBook, currentChar, {});
            persistReviewResults();
            markDirtyChar(currentBook, currentChar);
            saveToServer();
        }

        // 保存审查结果（修复版：不删除未在当前页的数据）
        function saveReview() {
            // 关键修复：
            // 之前的逻辑会删除"未选中"的实例，导致翻页时数据丢失
            // 新逻辑：只在 toggleInstance() 中修改数据，这里仅负责同步和更新显示

            // 保证内存状态已更新
            persistReviewResults();

            // 更新进度显示
            updateReviewProgress();

            // 注意：不能调用 loadBook()！
            // 因为 loadBook() 会调用 syncBookFromServer()，从服务器覆盖本地数据
            // 而用户刚选择的数据可能还没上传（30秒间隔），会导致数据丢失
            // 字符列表的"已审查"标记会在关闭模态框时更新
        }

        // ============== 数据操作函数 ==============

        // 获取字符的实例数据
        function getCharInstances(bookName, char) {
            if (!reviewResults.books || !reviewResults.books[bookName]) {
                return {};
            }
            const charData = reviewResults.books[bookName][char];
            return charData ? (charData.instances || {}) : {};
        }

        // 设置字符数据（带时间戳）
        function setCharData(bookName, char, instances) {
            if (!reviewResults.books) {
                reviewResults.books = {};
            }
            if (!reviewResults.books[bookName]) {
                reviewResults.books[bookName] = {};
            }

            reviewResults.books[bookName][char] = {
                instances: instances,
                timestamp: new Date().toISOString()
            };
        }

        // 更新单个实例（带时间戳更新）
        function updateCharInstance(bookName, char, index, selected) {
            if (!reviewResults.books) {
                reviewResults.books = {};
            }
            if (!reviewResults.books[bookName]) {
                reviewResults.books[bookName] = {};
            }
            if (!reviewResults.books[bookName][char]) {
                reviewResults.books[bookName][char] = {
                    instances: {},
                    timestamp: new Date().toISOString()
                };
            }

            // 更新实例
            if (selected) {
                reviewResults.books[bookName][char].instances[index] = true;
            } else {
                delete reviewResults.books[bookName][char].instances[index];
            }

            // 更新时间戳
            reviewResults.books[bookName][char].timestamp = new Date().toISOString();
        }

        async function syncBookFromServer(bookName) {
            return store.syncBookFromServer(bookName);
        }

        async function saveToServer() {
            return store.saveToServer();
        }

        // 关闭模态窗口（自动保存）
        function closeModal() {
            // 自动保存当前页的审查结果（快速同步到本地和内存）
            if (currentChar) {
                saveReview();
                // 异步上传，不阻塞关闭
                saveToServer();
            }

            document.getElementById('review-modal').classList.remove('active');
            currentChar = null;

            // 异步刷新字符列表，避免阻塞 UI
            if (currentBook) {
                refreshBookDisplay();
            }
        }

        // 导出审查结果
        function exportResults() {
            const dataStr = JSON.stringify(reviewResults, null, 2);
            const blob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const timestamp = new Date().toISOString().slice(0,19).replace(/:/g,'-');
            a.download = `review_results_${timestamp}.json`;
            a.click();
            URL.revokeObjectURL(url);
            alert('审查结果已导出！');
        }

        // 导入审查结果
        function importResults() {
            document.getElementById('import-file').click();
        }

        // 处理导入的文件
        function handleImport(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const imported = JSON.parse(e.target.result);
                    reviewResults = imported;
                    persistReviewResults();
                    alert('审查结果已导入！');

                    // 如果当前有打开的书籍，刷新进度显示
                    if (currentBook) {
                        updateReviewProgress();
                    }
                } catch (error) {
                    alert('导入失败：文件格式不正确');
                    console.error('导入失败:', error);
                }
            };
            reader.readAsText(file);
        }

        // 更新审查进度
        function updateReviewProgress() {
            if (!currentBook) return;

            const bookReview = reviewResults.books?.[currentBook] || {};
            const reviewedChars = Object.keys(bookReview).filter(char => {
                const charData = bookReview[char];
                // v2格式：检查 instances 是否有数据
                return charData && charData.instances && Object.keys(charData.instances).length > 0;
            });

            const totalChars = parseInt(document.getElementById('stat-chars').textContent) || 0;
            const reviewedCount = reviewedChars.length;
            const progress = totalChars > 0 ? Math.round((reviewedCount / totalChars) * 100) : 0;

            document.getElementById('stat-reviewed').textContent = reviewedCount;
            document.getElementById('stat-progress').textContent = progress + '%';
        }

        // 检查字符是否已审查（v2格式）
        function isCharReviewed(char) {
            if (!currentBook) return false;
            const instances = getCharInstances(currentBook, char);
            return instances && Object.keys(instances).length > 0;
        }
})();
