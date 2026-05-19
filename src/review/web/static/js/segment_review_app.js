(function () {
const API_BASE = window.ReviewApi.apiBase();

        let currentBook = null;
        let currentChar = null;
        let currentInstances = [];
        let currentInstanceIndex = 0;
        let segmentationData = {};  // 切割结果缓存
        let reviewStatus = {};      // 审查状态
        let standardCharsData = null;  // 标准字数据（84法分类）
        let markedInstances = {};   // 标记的实例
        let currentOriginalBbox = null;  // 当前实例的原始 bbox
        let currentCharElement = null;   // 当前选中的字符元素，用于滚动定位
        let currentCharOrder = [];       // 当前书籍的字符顺序（包含跨组重复）
        let visitedChars = new Set();    // 本次会话已处理过的字符（按字形去重）
        let pendingAutoChar = null;      // 兼容旧逻辑：按名称
        let pendingAutoIndex = null;     // 新逻辑：按当前 DOM 全局索引跳转
        let standardCharsPromise = null;
        let markedInstancesPromise = null;

        // Bbox Slider 调整相关变量
        let currentSliderBbox = {};       // 当前slider计算的bbox
        let roiShape = { width: 100, height: 100 };  // ROI图片尺寸
        let processedRoiImage = null;     // 处理后的ROI图片，用于bbox预览
        let bboxBaseCanvas = null;        // Bbox 预览基础图层
        let bboxOverlayCanvas = null;     // Bbox 辅助线图层
        let bboxOverlayCtx = null;        // 辅助线图层上下文

        // 画笔工具相关变量
        let brushCanvas = null;         // 画笔 Canvas 元素
        let brushCtx = null;            // 画笔 Canvas 上下文
        let brushColor = 'white';       // 当前画笔颜色（默认白色）
        let brushSize = 2;              // 当前画笔大小（单位：ROI 像素）
        let isDrawing = false;          // 是否正在绘制
        let brushHistory = [];          // 画笔历史（撤销）
        let brushHistoryStep = -1;     // 当前历史步骤
        let currentSegmentedImage = null;  // 当前切割结果图片的 base64
        let roiToPreviewScale = 1;         // ROI→预览 缩放比例（用于将 ROI 像素换算为预览画布像素）
        let isImageMagnified = false;     // 图片是否已放大150%
        const paramsManager = window.SegmentReviewParams.createManager({ apiBase: API_BASE });

        function notifyParentDirty() {
            if (window.self === window.top) return;
            try {
                window.top.postMessage({ type: 'segment-state-dirty' }, window.location.origin);
            } catch (_) {}
        }

        // ==================== 初始化 ====================

        function bindStaticControls() {
            document.getElementById('close-review-modal-btn').addEventListener('click', closeModal);
            document.getElementById('confirm-segmentation-btn').addEventListener('click', confirmSegmentation);
            document.getElementById('mark-not-needed-btn').addEventListener('click', markInstanceAsNotNeeded);
            document.getElementById('toggle-manual-adjust-btn').addEventListener('click', toggleManualAdjustPanel);
            document.getElementById('unconfirm-segmentation-btn').addEventListener('click', unconfirmSegmentation);
            document.getElementById('brush-black').addEventListener('click', () => setBrushColor('black'));
            document.getElementById('brush-white').addEventListener('click', () => setBrushColor('white'));
            document.getElementById('brush-size-slider').addEventListener('input', event => setBrushSize(event.target.value));
            document.getElementById('undo-brush-btn').addEventListener('click', undoBrush);
            document.getElementById('redo-brush-btn').addEventListener('click', redoBrush);
            document.getElementById('clear-brush-btn').addEventListener('click', clearBrush);
            document.getElementById('shrink-threshold-slider').addEventListener('input', event => {
                document.getElementById('shrink-threshold-value').textContent = event.target.value;
            });
            document.getElementById('shrink-padding-slider').addEventListener('input', event => {
                document.getElementById('shrink-padding-value').textContent = event.target.value;
            });
            document.getElementById('shrink-bbox-btn').addEventListener('click', shrinkBboxToInk);
            document.getElementById('apply-manual-changes-btn').addEventListener('click', applyManualChanges);
            document.getElementById('cancel-manual-adjustment-btn').addEventListener('click', cancelManualAdjustment);
            document.getElementById('prev-btn').addEventListener('click', loadPrevInstance);
            document.getElementById('next-btn').addEventListener('click', loadNextInstance);
            document.getElementById('apply-custom-params-btn').addEventListener('click', applyCustomParams);
            document.getElementById('reset-params-btn').addEventListener('click', resetParams);
        }

        document.addEventListener('DOMContentLoaded', async function() {
            try {
                bindStaticControls();
                const bookSelect = document.getElementById('book-select');
                if (bookSelect) {
                    bookSelect.disabled = true;
                }
                await Promise.all([
                    loadBooks(),
                    loadStandardChars(),
                    loadMarkedInstances(),
                ]);
                if (bookSelect) {
                    bookSelect.disabled = false;
                }
                await handleDeepLink();
            } catch (error) {
                console.error('初始化过程中出错:', error);
            }

            // 添加图片大小控制事件监听
            document.getElementById('image-size-slider').addEventListener('input', (e) => {
                updateImageSize(e.target.value);
            });

            // 添加键盘快捷键
            document.addEventListener('keydown', function(e) {
                const modal = document.getElementById('review-modal');
                const isModalOpen = modal.classList.contains('active');

                // ESC：优先关闭当前模态；若未打开模态则通知父级关闭 iframe 弹窗
                if (e.key === 'Escape') {
                    if (isModalOpen) {
                        closeModal();
                    } else if (window.self !== window.top) {
                        try {
                            window.top.postMessage({ type: 'close-segment-modal' }, window.location.origin);
                        } catch (_) {}
                    }
                    return;
                }

                if (!isModalOpen) return;

                // 左箭头: 上一个
                if (e.key === 'ArrowLeft') {
                    loadPrevInstance();
                }
                // 右箭头: 下一个
                else if (e.key === 'ArrowRight') {
                    loadNextInstance();
                }
                // Enter: 确认通过
                else if (e.key === 'Enter') {
                    confirmSegmentation();
                }
            });
        });

        async function handleDeepLink() {
            try {
                const params = new URLSearchParams(window.location.search);
                const book = params.get('book');
                const ch = params.get('char');
                const inst = params.get('instance_id');
                if (!book) return;

                // 选择书籍
                const sel = document.getElementById('book-select');
                if (sel) sel.value = book;
                await loadBook(book);

                if (!ch) return;
                // 打开字符
                await openCharByName(ch);

                if (inst && Array.isArray(currentInstances)) {
                    const idx = currentInstances.indexOf(inst);
                    if (idx >= 0) {
                        currentInstanceIndex = idx;
                        await loadInstance();
                    }
                }
            } catch (e) {
                console.warn('DeepLink 处理失败：', e);
            }
        }

        // ==================== 加载数据 ====================

        async function loadBooks() {
            try {
                const response = await fetch(`${API_BASE}/books_simple`);
                const data = await response.json();

                const select = document.getElementById('book-select');
                select.innerHTML = '<option value="">-- 请选择书籍 --</option>';

                data.books.forEach(book => {
                    const option = document.createElement('option');
                    option.value = book.name;
                    option.textContent = book.name;
                    select.appendChild(option);
                });

                select.addEventListener('change', function() {
                    if (this.value) {
                        loadBook(this.value);
                    }
                });
            } catch (error) {
                console.error('加载书籍列表失败:', error);
                alert('加载书籍列表失败');
            }
        }

        async function loadStandardChars() {
            if (!standardCharsPromise) {
                standardCharsPromise = (async () => {
                    try {
                        const response = await fetch(`${API_BASE}/characters_structure`);
                        const data = await response.json();
                        standardCharsData = data.structure;
                        return standardCharsData;
                    } catch (error) {
                        console.error('加载标准字数据失败:', error);
                        standardCharsData = null;
                        throw error;
                    }
                })();
            }
            return standardCharsPromise;
        }

        async function loadMarkedInstances() {
            if (!markedInstancesPromise) {
                markedInstancesPromise = (async () => {
                    try {
                        const response = await fetch(`${API_BASE}/marked_instances`);
                        const data = await response.json();
                        if (data.success) {
                            markedInstances = data.marked_instances || {};
                        }
                        return markedInstances;
                    } catch (error) {
                        console.error('加载标记实例失败:', error);
                        markedInstances = {};
                        throw error;
                    }
                })();
            }
            return markedInstancesPromise;
        }

        async function loadBook(bookName, options = {}) {
            const preserveVisited = !!options.preserveVisited;
            currentBook = bookName;
            document.getElementById('char-list').innerHTML = '<div class="spinner"></div>';
            currentCharOrder = [];
            if (!preserveVisited) {
                visitedChars = new Set();
            }

            try {
                await loadStandardChars();
                await loadMarkedInstances();

                // 从 lookup 文件获取字符列表和统计信息（O(1) 查询）
                const response = await fetch(`${API_BASE}/segment_book_chars?book=${encodeURIComponent(bookName)}`);
                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || '加载失败');
                }

                const charsInfo = data.chars;

                // 统计信息
                let totalChars = 0;
                let confirmedCount = 0;
                let unreviewedCount = 0;

                const charList = document.getElementById('char-list');
                charList.innerHTML = '';

                // 按 84 法分组
                if (!standardCharsData) {
                    charList.innerHTML = '<p style="text-align: center; color: #999;">标准字数据加载失败</p>';
                    return;
                }

                for (const [groupName, chars] of Object.entries(standardCharsData)) {
                    const groupDiv = document.createElement('div');
                    groupDiv.className = 'char-group';

                    const header = document.createElement('div');
                    header.className = 'char-group-header';
                    header.innerHTML = `
                        <span class="char-group-title">${groupName}</span>
                        <span class="char-group-toggle">▼</span>
                    `;
                    header.onclick = () => groupDiv.classList.toggle('collapsed');

                    const content = document.createElement('div');
                    content.className = 'char-group-content';

                    chars.forEach(char => {
                        // 检查该字符是否在 lookup 文件中（已通过第一轮审查）
                        const charInfo = charsInfo[char];
                        const totalForChar = (charInfo && typeof charInfo.total === 'number') ? charInfo.total : 0;
                        if (!charInfo || totalForChar === 0) return;

                        totalChars++;

                        const confirmed = charInfo.confirmed || 0;
                        const unreviewed = charInfo.count - confirmed;

                        confirmedCount += confirmed;
                        unreviewedCount += unreviewed;

                        // 计算标记数量
                        let markCount = 0;
                        const instanceKeyPrefix = `${bookName}_${char}_`;
                        for (const key in markedInstances) {
                            if (key.startsWith(instanceKeyPrefix)) {
                                markCount++;
                            }
                        }

                        const charItem = document.createElement('div');
                        charItem.className = 'char-item';
                        charItem.style.position = 'relative';

                        if (confirmed === charInfo.count) {
                            charItem.classList.add('completed');
                        } else if (confirmed > 0) {
                            charItem.classList.add('partial');
                        }

                        charItem.innerHTML = `
                            <div class="char-text">${char}</div>
                            <div class="char-status">${confirmed}/${charInfo.count}</div>
                            ${markCount > 0 ? `<div class="char-mark-count">${markCount}</div>` : ''}
                        `;

                        charItem.onclick = () => openCharReview(char, charItem);

                        content.appendChild(charItem);
                        currentCharOrder.push(char);
                    });

                    if (content.children.length > 0) {
                        groupDiv.appendChild(header);
                        groupDiv.appendChild(content);
                        charList.appendChild(groupDiv);
                    }
                }

                // 更新统计
                document.getElementById('stat-total').textContent = totalChars;
                document.getElementById('stat-confirmed').textContent = confirmedCount;
                document.getElementById('stat-unreviewed').textContent = unreviewedCount;

                if (charList.children.length === 0) {
                    charList.innerHTML = '<p style="text-align: center; color: #999; padding: 40px 0;">该书籍当前没有 filter accepted 的字符</p>';
                }
            } catch (error) {
                console.error('加载书籍数据失败:', error);
                const message = error && error.message ? error.message : String(error);
                document.getElementById('char-list').innerHTML = `<p style="text-align: center; color: #f44336; padding: 40px 0;">加载失败: ${message}</p>`;
            }
        }

        // ==================== 打开字符审查 ====================

        async function openCharReview(char, charElement) {
            currentChar = char;
            currentCharElement = charElement;  // 保存字符元素引用，用于关闭时滚动定位
            // 标记该字符已访问，用于自动跳过跨组重复
            try { visitedChars.add(char); } catch (e) {}

            try {
                // 从 lookup 文件获取该字符的所有 instance_id（O(1) 查询）
                const response = await fetch(`${API_BASE}/segment_instance_ids?book=${encodeURIComponent(currentBook)}&char=${encodeURIComponent(char)}`);
                const data = await response.json();

                if (!data.success) {
                    alert(data.error || '加载实例列表失败');
                    return;
                }

                currentInstances = data.instance_ids;

                if (currentInstances.length === 0) {
                    alert('该字符当前没有 filter accepted 的实例');
                    return;
                }

                // 打开模态窗口
                document.getElementById('review-modal').classList.add('active');
                // 锁定 body 滚动，防止滚动穿透
                document.body.style.overflow = 'hidden';
                currentInstanceIndex = 0;

                // 初始化参数面板（如果还没有初始化）
                initParamsPanel();

                await loadInstance();

            } catch (error) {
                console.error('打开字符审查失败:', error);
                alert('加载失败');
            }
        }

        // ==================== 加载实例 ====================

        async function loadInstance() {
            if (currentInstanceIndex < 0 || currentInstanceIndex >= currentInstances.length) {
                return;
            }

            const instanceId = currentInstances[currentInstanceIndex];
            document.getElementById('modal-title').textContent = `审查 "${currentChar}" - ${instanceId}`;
            document.getElementById('page-info').textContent = `${currentInstanceIndex + 1} / ${currentInstances.length}`;

            // 更新分页按钮状态
            document.getElementById('prev-btn').disabled = currentInstanceIndex === 0;
            document.getElementById('next-btn').disabled = currentInstanceIndex === currentInstances.length - 1;

            // 显示加载状态，重置图片大小
            document.getElementById('instance-info').innerHTML = '<span class="status-badge loading">加载中...</span>';
            ['roi-container', 'segmented-container', 'debug-container'].forEach(id => {
                const container = document.getElementById(id);
                container.innerHTML = '<span class="loading-text">加载中...</span>';
                container.style.transform = 'scale(1)';
                container.classList.remove('scaled');
            });

            // 重置图片大小控制（默认 300%）
            document.getElementById('image-size-slider').value = 300;
            document.getElementById('image-size-value').textContent = '300';

            // 重置 bbox 预览图片大小
            [document.getElementById('bbox-base-canvas'), document.getElementById('bbox-overlay-canvas'), document.getElementById('brush-canvas')].forEach(canvasEl => {
                if (canvasEl) {
                    canvasEl.style.transform = 'scale(1)';
                }
            });

            try {
                // 请求切割
                
                const response = await fetch(`${API_BASE}/segment_instances`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        instance_id: instanceId
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error);
                }

                // 缓存结果
                segmentationData[instanceId] = data;

                if (data.review_entry) {
                    if (!reviewStatus[currentChar]) reviewStatus[currentChar] = {};
                    reviewStatus[currentChar][instanceId] = {
                        status: data.review_entry.status || 'unreviewed',
                        method: data.review_entry.method,
                        decision: data.review_entry.decision || 'unknown'
                    };
                }

                // 保存 segmented bbox（切割结果）和ROI形状
                if (data.metadata && data.metadata.segmented_bbox) {
                    currentSliderBbox = { ...data.metadata.segmented_bbox };
                    // 同时保存作为参考的原始值（用于取消时恢复）
                    if (data.metadata && data.metadata.original_bbox) {
                        currentOriginalBbox = { ...data.metadata.original_bbox };
                    }
                }
                if (data.metadata && data.metadata.roi_shape) {
                    roiShape = {
                        width: data.metadata.roi_shape[1],
                        height: data.metadata.roi_shape[0]
                    };
                }

                // 显示图片
                document.getElementById('roi-container').innerHTML = `<img src="${data.roi_image}" alt="原始ROI">`;
                document.getElementById('segmented-container').innerHTML = `<img src="${data.segmented_image}" alt="切割结果">`;
                document.getElementById('debug-container').innerHTML = `<img src="${data.debug_image}" alt="调试图">`;

                // 重新应用当前的图片缩放设置
                const currentScale = document.getElementById('image-size-slider').value;
                updateImageSize(currentScale);

                // 保存processed_roi用于bbox预览（处理后的ROI图片）
                if (data.processed_roi) {
                    processedRoiImage = data.processed_roi;
                } else {
                    processedRoiImage = null;
                }

                // 如果slider面板处于活动状态，重新初始化
                const sliderPanel = document.getElementById('bbox-slider-panel');
                if (sliderPanel.classList.contains('active')) {
                    initializeBboxSlider();
                }

                // 更新状态信息
                const statusBadge = reviewStatus[currentChar]?.[instanceId];
                const decision = statusBadge?.decision || 'unknown';
                const badgeStatus = statusBadge?.status || 'unreviewed';

                let statusHtml = '<span class="status-badge">未审查</span>';
                if (decision === 'drop' || badgeStatus === 'dropped') {
                    statusHtml = '<span class="status-badge warning">已标记不需要</span>';
                } else if (badgeStatus === 'confirmed') {
                    statusHtml = '<span class="status-badge success">已确认</span>';
                }
                document.getElementById('instance-info').innerHTML = statusHtml;

                if (decision === 'drop' || badgeStatus === 'dropped') {
                    const segContainer = document.getElementById('segmented-container');
                    const imgEl = segContainer ? segContainer.querySelector('img') : null;
                    if (imgEl) {
                        imgEl.classList.add('not-needed-image');
                    } else if (segContainer) {
                        segContainer.innerHTML = '<div class="not-needed-placeholder">该实例已被标记为不需要</div>';
                    }
                }

            } catch (error) {
                console.error('加载实例失败:', error);
                document.getElementById('instance-info').innerHTML = `<span class="status-badge error">加载失败: ${error.message}</span>`;
            }
        }

        function loadPrevInstance() {
            if (currentInstanceIndex > 0) {
                currentInstanceIndex--;
                loadInstance();
            }
        }

        function loadNextInstance() {
            if (currentInstanceIndex < currentInstances.length - 1) {
                currentInstanceIndex++;
                loadInstance();
            }
        }

        function getNextCharIndex() {
            // 基于当前字符元素在 DOM 中的位置，寻找后续第一个未访问的字符索引
            const items = document.querySelectorAll('.char-item');
            if (!items || items.length === 0 || !currentCharElement) return null;

            let passedCurrent = false;
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                if (!passedCurrent) {
                    if (item === currentCharElement) passedCurrent = true;
                    continue;
                }
                const textEl = item.querySelector('.char-text');
                const name = textEl ? textEl.textContent : null;
                if (!name) continue;
                if (visitedChars && visitedChars.has(name)) continue;
                return i;
            }
            return null;
        }

        function openCharByIndex(index) {
            const items = document.querySelectorAll('.char-item');
            if (!items || index == null) return;
            if (index < 0 || index >= items.length) return;
            const item = items[index];
            const textEl = item.querySelector('.char-text');
            const name = textEl ? textEl.textContent : null;
            if (name) {
                openCharReview(name, item);
            }
        }

        async function openCharByName(charName) {
            if (!charName) return;
            const charItems = document.querySelectorAll('.char-item');
            for (const item of charItems) {
                const textEl = item.querySelector('.char-text');
                if (textEl && textEl.textContent === charName) {
                    await openCharReview(charName, item);
                    return;
                }
            }
        }

        // ==================== 操作功能 ====================

        async function confirmSegmentation() {
            const instanceId = currentInstances[currentInstanceIndex];
            const data = segmentationData[instanceId];

            if (!data) {
                alert('请先加载实例');
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/save_segmentation`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        instance_id: instanceId,
                        status: 'confirmed',
                        method: 'auto',
                        segmented_image_base64: data.segmented_image,
                        decision: 'need'
                    })
                });

                const result = await response.json();
                if (!result.success) {
                    throw new Error(result.error);
                }

                // 更新状态
                if (!reviewStatus[currentChar]) reviewStatus[currentChar] = {};
                reviewStatus[currentChar][instanceId] = { status: 'confirmed', decision: 'need' };
                notifyParentDirty();

                if (currentInstanceIndex < currentInstances.length - 1) {
                    loadNextInstance();
                } else {
                    // 记录基于当前元素位置的下一个字符索引，避免跨组重复导致跳回
                    pendingAutoIndex = getNextCharIndex();
                    pendingAutoChar = null; // 不再按名称自动跳转
                    await closeModal();
                }

            } catch (error) {
                console.error('保存失败:', error);
                alert('保存失败: ' + error.message);
            }
        }

        async function unconfirmSegmentation() {
            const instanceId = currentInstances[currentInstanceIndex];
            if (!confirm('确认取消通过该实例吗？')) return;

            try {
                const response = await fetch(`${API_BASE}/unconfirm_segmentation`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        instance_id: instanceId
                    })
                });
                const ct = (response.headers.get('content-type') || '').toLowerCase();
                let result = null;
                if (ct.includes('application/json')) {
                    result = await response.json();
                } else {
                    const text = await response.text();
                    throw new Error(text || `HTTP ${response.status}`);
                }
                if (!response.ok || !result.success) throw new Error((result && result.error) || '取消失败');

                // 更新前端状态与徽标
                if (!reviewStatus[currentChar]) reviewStatus[currentChar] = {};
                reviewStatus[currentChar][instanceId] = { status: 'unreviewed', decision: 'unknown' };
                notifyParentDirty();
                document.getElementById('instance-info').innerHTML = '<span class="status-badge">未审查</span>';

                // 刷新外层统计
                await loadBook(currentBook, { preserveVisited: true });

            } catch (e) {
                alert('取消失败: ' + e.message);
            }
        }

        async function markInstanceAsNotNeeded() {
            const instanceId = currentInstances[currentInstanceIndex];
            if (!confirm('确认将该实例标记为“不需要”？')) return;
            try {
                const response = await fetch(`${API_BASE}/mark_segmentation_decision`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        instance_id: instanceId,
                        decision: 'drop'
                    })
                });
                const result = await response.json();
                if (!response.ok || !result.success) throw new Error(result.error || '标记失败');

                if (!reviewStatus[currentChar]) reviewStatus[currentChar] = {};
                reviewStatus[currentChar][instanceId] = { status: 'dropped', decision: 'drop' };
                notifyParentDirty();
                document.getElementById('instance-info').innerHTML = '<span class="status-badge warning">已标记不需要</span>';
                const segContainer = document.getElementById('segmented-container');
                const imgEl = segContainer ? segContainer.querySelector('img') : null;
                if (imgEl) {
                    imgEl.classList.add('not-needed-image');
                } else if (segContainer) {
                    segContainer.innerHTML = '<div class="not-needed-placeholder">该实例已标记为不需要</div>';
                }
                await loadBook(currentBook, { preserveVisited: true });
            } catch (e) {
                alert('标记失败: ' + e.message);
            }
        }

        // 图片大小控制功能
        function updateImageSize(scale) {
            document.getElementById('image-size-value').textContent = scale;

            // 更新各种图片容器（不包括调试可视化）
            ['roi-container', 'segmented-container'].forEach(id => {
                const container = document.getElementById(id);
                if (container) {
                    container.style.transform = `scale(${scale/100})`;
                    container.style.transformOrigin = 'center';
                    container.classList.add('scaled');
                }
            });

            // 更新 bbox 预览图片
            const baseCanvas = document.getElementById('bbox-base-canvas');
            const overlayCanvas = document.getElementById('bbox-overlay-canvas');
            const brushCanvasEl = document.getElementById('brush-canvas');
            [baseCanvas, overlayCanvas, brushCanvasEl].forEach(canvasEl => {
                if (canvasEl) {
                    canvasEl.style.transform = `translate(-50%, -50%) scale(${scale/100})`;
                    canvasEl.style.transformOrigin = 'center';
                }
            });
        }

        // 在模态窗口打开时自动初始化参数面板
        function initParamsPanel() {
            return paramsManager.initPanel();
        }

        // Bbox 调整功能
    
        // Bbox Slider 调整相关函数
        function initializeBboxSlider() {
            if (!currentSliderBbox || !roiShape) return;

            // 设置slider的最大值
            document.getElementById('bbox-left-slider').max = roiShape.width;
            document.getElementById('bbox-right-slider').max = roiShape.width;
            document.getElementById('bbox-top-slider').max = roiShape.height;
            document.getElementById('bbox-bottom-slider').max = roiShape.height;

            // 设置初始值 - 使用 segmented bbox
            const left = currentSliderBbox.x;
            const right = currentSliderBbox.x + currentSliderBbox.width;
            const top = currentSliderBbox.y;
            const bottom = currentSliderBbox.y + currentSliderBbox.height;

            document.getElementById('bbox-left-slider').value = left;
            document.getElementById('bbox-right-slider').value = right;
            document.getElementById('bbox-top-slider').value = top;
            document.getElementById('bbox-bottom-slider').value = bottom;

            // 更新显示值
            updateSliderValues();

            // 添加实时更新事件监听
            ['bbox-left-slider', 'bbox-right-slider', 'bbox-top-slider', 'bbox-bottom-slider'].forEach(id => {
                const slider = document.getElementById(id);
                slider.removeEventListener('input', onSliderChange);
                slider.addEventListener('input', onSliderChange);
            });

            // 初始化预览
            updateBboxPreview();
        }

        function onSliderChange() {
            updateSliderValues();
            validateSliderConstraints();
            // 实时更新预览
            updateBboxPreview();
        }

        // 收边：在当前 bbox 内自动收缩到墨迹（黑色）区域
        function shrinkBboxToInk() {
            try {
                if (!currentSliderBbox || !roiShape) {
                    alert('预览未就绪');
                    return;
                }

                const baseCanvas = document.getElementById('bbox-base-canvas');
                if (!baseCanvas || baseCanvas.width === 0 || baseCanvas.height === 0) {
                    alert('预览画布未就绪');
                    return;
                }

                // 使用 ROI 尺寸进行计算，确保坐标精确
                const baseSrc = processedRoiImage || currentSegmentedImage;
                if (!baseSrc) {
                    alert('缺少底图，无法收边');
                    return;
                }

                // 1) 加载 ROI 尺寸的底图
                const img = new Image();
                img.onload = () => {
                    const roiW = img.width;
                    const roiH = img.height;

                    // 2) 将画笔层放大到 ROI 尺寸
                    const brushLayer = document.getElementById('brush-canvas');
                    const brushOnRoi = document.createElement('canvas');
                    brushOnRoi.width = roiW;
                    brushOnRoi.height = roiH;
                    const brushRoiCtx = brushOnRoi.getContext('2d');
                    if (brushLayer && brushLayer.width > 0 && brushLayer.height > 0) {
                        const scaleX = roiW / baseCanvas.width;
                        const scaleY = roiH / baseCanvas.height;
                        brushRoiCtx.drawImage(
                            brushLayer,
                            0, 0, brushLayer.width, brushLayer.height,
                            0, 0, Math.round(brushLayer.width * scaleX), Math.round(brushLayer.height * scaleY)
                        );
                    }

                    // 3) 合成 ROI 图（底图 + 画笔）
                    const comp = document.createElement('canvas');
                    comp.width = roiW;
                    comp.height = roiH;
                    const compCtx = comp.getContext('2d', { willReadFrequently: true });
                    compCtx.drawImage(img, 0, 0);
                    compCtx.drawImage(brushOnRoi, 0, 0);

                    // 4) 在当前 bbox 内查找“黑色”像素的最小外接矩形
                    const bbox = currentSliderBbox;
                    const sx = Math.max(0, Math.floor(bbox.x));
                    const sy = Math.max(0, Math.floor(bbox.y));
                    const sw = Math.max(1, Math.min(roiW - sx, Math.floor(bbox.width)));
                    const sh = Math.max(1, Math.min(roiH - sy, Math.floor(bbox.height)));

                    let top = sh, left = sw, bottom = -1, right = -1;
                    const thrEl = document.getElementById('shrink-threshold-slider');
                    const padEl = document.getElementById('shrink-padding-slider');
                    const threshold = thrEl ? Math.max(0, Math.min(255, parseInt(thrEl.value || '220', 10))) : 220;
                    const pad = padEl ? Math.max(0, Math.min(20, parseInt(padEl.value || '1', 10))) : 1;

                    const imgData = compCtx.getImageData(sx, sy, sw, sh);
                    const data = imgData.data;
                    for (let y = 0; y < sh; y++) {
                        for (let x = 0; x < sw; x++) {
                            const idx = (y * sw + x) * 4;
                            const r = data[idx], g = data[idx + 1], b = data[idx + 2], a = data[idx + 3];
                            if (a === 0) continue;
                            const gray = (r + g + b) / 3;
                            if (gray <= threshold) {
                                if (x < left) left = x;
                                if (x > right) right = x;
                                if (y < top) top = y;
                                if (y > bottom) bottom = y;
                            }
                        }
                    }

                    if (bottom === -1) {
                        alert('未检测到墨迹，已保持不变');
                        return;
                    }

                    left = Math.max(0, left - pad);
                    top = Math.max(0, top - pad);
                    right = Math.min(sw - 1, right + pad);
                    bottom = Math.min(sh - 1, bottom + pad);

                    const newX = sx + left;
                    const newY = sy + top;
                    const newW = right - left + 1;
                    const newH = bottom - top + 1;

                    // 5) 写回 slider 并刷新预览
                    const leftSlider = document.getElementById('bbox-left-slider');
                    const rightSlider = document.getElementById('bbox-right-slider');
                    const topSlider = document.getElementById('bbox-top-slider');
                    const bottomSlider = document.getElementById('bbox-bottom-slider');

                    leftSlider.value = newX;
                    rightSlider.value = newX + newW;
                    topSlider.value = newY;
                    bottomSlider.value = newY + newH;

                    updateSliderValues();
                    validateSliderConstraints();
                    updateBboxPreview();

                };
                img.onerror = () => alert('底图加载失败，无法收边');
                img.src = baseSrc;

            } catch (e) {
                console.error('收边失败:', e);
                alert('收边失败: ' + e.message);
            }
        }

        function loadImageElement(src) {
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = reject;
                img.src = src;
            });
        }

        // 绘制bbox边界框预览函数 - 优先使用 processed ROI
        function updateBboxPreview() {
            if (!currentSliderBbox) {
                return;
            }

            const container = document.querySelector('.bbox-preview-container');
            if (!container) return;

            if (!bboxBaseCanvas) bboxBaseCanvas = document.getElementById('bbox-base-canvas');
            if (!bboxOverlayCanvas) bboxOverlayCanvas = document.getElementById('bbox-overlay-canvas');
            if (!bboxBaseCanvas || !bboxOverlayCanvas) return;

            bboxOverlayCtx = bboxOverlayCanvas.getContext('2d');
            if (!brushCanvas) {
                brushCanvas = document.getElementById('brush-canvas');
                if (brushCanvas) {
                    brushCtx = brushCanvas.getContext('2d');
                }
            }

            const segmentedContainer = document.getElementById('segmented-container');
            const segmentedImg = segmentedContainer ? segmentedContainer.querySelector('img') : null;

            let imageSource = null;
            if (processedRoiImage) {
                imageSource = processedRoiImage;
            } else if (segmentedImg && segmentedImg.src) {
                imageSource = segmentedImg.src;
            }

            if (!imageSource) {
                return;
            }

            const img = new Image();
            img.onload = function() {
                const maxWidth = 500;
                const maxHeight = 350;

                let width = img.width;
                let height = img.height;

                const scale = Math.min(maxWidth / width, maxHeight / height, 1);
                width *= scale;
                height *= scale;
                // 记录 ROI→预览 比例，便于将 ROI 像素的笔宽换算到预览画布
                roiToPreviewScale = scale;

                [bboxBaseCanvas, bboxOverlayCanvas, brushCanvas].forEach(canvasEl => {
                    if (!canvasEl) return;
                    canvasEl.width = width;
                    canvasEl.height = height;
                    canvasEl.style.width = `${width}px`;
                    canvasEl.style.height = `${height}px`;
                    canvasEl.style.display = 'block';
                });

                const baseCtx = bboxBaseCanvas.getContext('2d');
                baseCtx.clearRect(0, 0, width, height);
                baseCtx.drawImage(img, 0, 0, width, height);

                if (bboxOverlayCtx) {
                    bboxOverlayCtx.clearRect(0, 0, width, height);
                }
                if (brushCtx) {
                    brushCtx.clearRect(0, 0, width, height);
                    brushHistory = [];
                    brushHistoryStep = -1;
                }

                const left = currentSliderBbox.x * scale;
                const top = currentSliderBbox.y * scale;
                const bboxWidth = currentSliderBbox.width * scale;
                const bboxHeight = currentSliderBbox.height * scale;

                if (bboxOverlayCtx) {
                    bboxOverlayCtx.strokeStyle = '#ff4444';
                    bboxOverlayCtx.lineWidth = 1;
                    bboxOverlayCtx.setLineDash([6, 4]);
                    bboxOverlayCtx.strokeRect(left, top, bboxWidth, bboxHeight);
                    bboxOverlayCtx.fillStyle = 'rgba(255, 68, 68, 0.08)';
                    bboxOverlayCtx.fillRect(left, top, bboxWidth, bboxHeight);

                    bboxOverlayCtx.setLineDash([]);
                    bboxOverlayCtx.fillStyle = '#ff4444';
                    const cornerSize = 5;
                    bboxOverlayCtx.fillRect(left - cornerSize/2, top - cornerSize/2, cornerSize, cornerSize);
                    bboxOverlayCtx.fillRect(left + bboxWidth - cornerSize/2, top - cornerSize/2, cornerSize, cornerSize);
                    bboxOverlayCtx.fillRect(left - cornerSize/2, top + bboxHeight - cornerSize/2, cornerSize, cornerSize);
                    bboxOverlayCtx.fillRect(left + bboxWidth - cornerSize/2, top + bboxHeight - cornerSize/2, cornerSize, cornerSize);
                }

                currentSegmentedImage = imageSource;

                // 确保画笔画布在底图尺寸就绪后正确初始化
                try {
                    initBrushCanvasOnBbox();
                } catch (e) {
                    console.warn('[Brush] init after preview failed:', e);
                }
            };

            img.src = imageSource;
        }

        function updateSliderValues() {
            const left = parseInt(document.getElementById('bbox-left-slider').value);
            const right = parseInt(document.getElementById('bbox-right-slider').value);
            const top = parseInt(document.getElementById('bbox-top-slider').value);
            const bottom = parseInt(document.getElementById('bbox-bottom-slider').value);

            // 更新显示值
            document.getElementById('bbox-left-value').textContent = left;
            document.getElementById('bbox-right-value').textContent = right;
            document.getElementById('bbox-top-value').textContent = top;
            document.getElementById('bbox-bottom-value').textContent = bottom;

            // 计算bbox
            const x = left;
            const y = top;
            const width = right - left;
            const height = bottom - top;

            // 更新bbox信息显示
            document.getElementById('bbox-x').textContent = x;
            document.getElementById('bbox-y').textContent = y;
            document.getElementById('bbox-width').textContent = width;
            document.getElementById('bbox-height').textContent = height;

            // 更新当前slider bbox
            currentSliderBbox = { x, y, width, height };
        }

        function validateSliderConstraints() {
            let isValid = true;
            let errors = {};

            const left = parseInt(document.getElementById('bbox-left-slider').value);
            const right = parseInt(document.getElementById('bbox-right-slider').value);
            const top = parseInt(document.getElementById('bbox-top-slider').value);
            const bottom = parseInt(document.getElementById('bbox-bottom-slider').value);

            // 清除所有错误信息
            ['bbox-left-error', 'bbox-right-error', 'bbox-top-error', 'bbox-bottom-error'].forEach(id => {
                document.getElementById(id).textContent = '';
            });

            // 检查边界约束
            if (left >= right) {
                document.getElementById('bbox-left-error').textContent = '左边界必须在右边界左侧';
                document.getElementById('bbox-right-error').textContent = '右边界必须在左边界右侧';
                isValid = false;
            }

            if (top >= bottom) {
                document.getElementById('bbox-top-error').textContent = '上边界必须在下边界上方';
                document.getElementById('bbox-bottom-error').textContent = '下边界必须在上边界下方';
                isValid = false;
            }

            // 检查最小尺寸
            if (right - left < 10) {
                document.getElementById('bbox-right-error').textContent = '宽度至少10像素';
                isValid = false;
            }

            if (bottom - top < 10) {
                document.getElementById('bbox-bottom-error').textContent = '高度至少10像素';
                isValid = false;
            }

            // 检查不超出图片范围
            if (left < 0 || right > roiShape.width) {
                if (left < 0) document.getElementById('bbox-left-error').textContent = '超出左边界';
                if (right > roiShape.width) document.getElementById('bbox-right-error').textContent = '超出右边界';
                isValid = false;
            }

            if (top < 0 || bottom > roiShape.height) {
                if (top < 0) document.getElementById('bbox-top-error').textContent = '超出上边界';
                if (bottom > roiShape.height) document.getElementById('bbox-bottom-error').textContent = '超出下边界';
                isValid = false;
            }

            return isValid;
        }

        async function applyBboxSliderAdjustment(options = {}) {
            const { silent = false } = options;

            if (!validateSliderConstraints()) {
                if (silent) {
                    alert('当前 Bbox 设置不合法，请检查边界。');
                }
                console.error('[Bbox Adjustment] 验证失败');
                return false;
            }

            const instanceId = currentInstances[currentInstanceIndex];
            const left = parseInt(document.getElementById('bbox-left-slider').value, 10);
            const right = parseInt(document.getElementById('bbox-right-slider').value, 10);
            const top = parseInt(document.getElementById('bbox-top-slider').value, 10);
            const bottom = parseInt(document.getElementById('bbox-bottom-slider').value, 10);

            const adjustedBbox = {
                x: left,
                y: top,
                width: right - left,
                height: bottom - top
            };

            // debug log removed

            const entry = segmentationData[instanceId] || {};

            try {
                if (processedRoiImage) {
                    const img = await loadImageElement(processedRoiImage);

                    const canvas = document.createElement('canvas');
                    canvas.width = adjustedBbox.width;
                    canvas.height = adjustedBbox.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(
                        img,
                        adjustedBbox.x,
                        adjustedBbox.y,
                        adjustedBbox.width,
                        adjustedBbox.height,
                        0,
                        0,
                        adjustedBbox.width,
                        adjustedBbox.height
                    );

                    const segmentedDataUrl = canvas.toDataURL('image/png');

                    entry.segmented_image = segmentedDataUrl;
                    entry.debug_image = segmentedDataUrl;
                    entry.processed_roi = segmentedDataUrl;
                    entry.metadata = entry.metadata || {};
                    entry.metadata.segmented_bbox = {
                        x: 0,
                        y: 0,
                        width: adjustedBbox.width,
                        height: adjustedBbox.height
                    };
                    entry.metadata.roi_shape = [adjustedBbox.height, adjustedBbox.width, 3];
                    segmentationData[instanceId] = entry;

                    processedRoiImage = segmentedDataUrl;
                    currentSegmentedImage = segmentedDataUrl;
                    currentSliderBbox = { x: 0, y: 0, width: adjustedBbox.width, height: adjustedBbox.height };
                    roiShape = { width: adjustedBbox.width, height: adjustedBbox.height };

                    document.getElementById('roi-container').innerHTML = `<img src="${segmentedDataUrl}" alt="原始ROI">`;
                    document.getElementById('segmented-container').innerHTML = `<img src="${segmentedDataUrl}" alt="切割结果">`;
                    document.getElementById('debug-container').innerHTML = `<img src="${segmentedDataUrl}" alt="调试图">`;

                    const currentScale = document.getElementById('image-size-slider').value;
                    updateImageSize(currentScale);

                    if (!silent) {
                        document.getElementById('bbox-slider-panel').classList.remove('active');
                        document.getElementById('instance-info').innerHTML = '<span class="status-badge success">Bbox 调整完成</span>';
                    }

                    updateBboxPreview();
                    return true;
                }

                const response = await fetch(`${API_BASE}/adjust_bbox`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        instance_id: instanceId,
                        adjusted_bbox: adjustedBbox
                    })
                });

                const data = await response.json();
                if (!data.success) {
                    throw new Error(data.error || '未知错误');
                }

                entry.segmented_image = data.segmented_image;
                entry.debug_image = data.debug_image;
                entry.processed_roi = data.processed_roi || null;
                entry.metadata = data.metadata || entry.metadata || {};
                segmentationData[instanceId] = entry;

                processedRoiImage = data.processed_roi || data.segmented_image;
                currentSegmentedImage = processedRoiImage || data.segmented_image;

                if (entry.metadata && entry.metadata.segmented_bbox) {
                    currentSliderBbox = { ...entry.metadata.segmented_bbox };
                } else {
                    currentSliderBbox = { ...adjustedBbox };
                }

                if (entry.metadata && entry.metadata.roi_shape) {
                    roiShape = {
                        width: entry.metadata.roi_shape[1],
                        height: entry.metadata.roi_shape[0]
                    };
                }

                document.getElementById('segmented-container').innerHTML = `<img src="${data.segmented_image}" alt="切割结果">`;
                document.getElementById('debug-container').innerHTML = `<img src="${data.debug_image}" alt="调试图">`;

                const currentScale = document.getElementById('image-size-slider').value;
                updateImageSize(currentScale);

                if (!silent) {
                    document.getElementById('bbox-slider-panel').classList.remove('active');
                    document.getElementById('instance-info').innerHTML = '<span class="status-badge success">Bbox 调整完成</span>';
                }

                updateBboxPreview();
                return true;

            } catch (error) {
                console.error('[Bbox Adjustment] 调整失败:', error);
                if (!silent) {
                    alert('Bbox 调整失败: ' + error.message);
                }
                return false;
            }
        }

        function cancelBboxSliderAdjustment(suppressClose = false) {
            const instanceId = currentInstances[currentInstanceIndex];
            const data = segmentationData[instanceId];

            if (data && data.metadata && data.metadata.segmented_bbox) {
                currentSliderBbox = { ...data.metadata.segmented_bbox };
            } else {
                currentSliderBbox = { ...currentOriginalBbox };
            }

            initializeBboxSlider();
            updateBboxPreview();

            if (!suppressClose) {
                document.getElementById('bbox-slider-panel').classList.remove('active');
            }
        }

        async function applyManualChanges() {
            try {
                // 需要 processedRoiImage 或当前切割图作为底图
                const baseSrc = processedRoiImage || currentSegmentedImage;
                if (!baseSrc) {
                    alert('当前没有可应用的底图');
                    return;
                }

                // 读取原始 ROI 尺寸的底图
                const roiImg = await loadImageElement(baseSrc);

                // 计算从预览到原始 ROI 的缩放比例
                const previewBase = document.getElementById('bbox-base-canvas');
                let scaleX = 1, scaleY = 1;
                if (previewBase && previewBase.width > 0 && previewBase.height > 0) {
                    scaleX = roiImg.width / previewBase.width;
                    scaleY = roiImg.height / previewBase.height;
                }

                // 将画笔层放大到 ROI 原始分辨率
                const brushLayer = document.getElementById('brush-canvas');
                const brushOnRoi = document.createElement('canvas');
                brushOnRoi.width = roiImg.width;
                brushOnRoi.height = roiImg.height;
                const brushOnRoiCtx = brushOnRoi.getContext('2d');
                if (brushLayer && brushLayer.width > 0 && brushLayer.height > 0) {
                    brushOnRoiCtx.drawImage(
                        brushLayer,
                        0, 0, brushLayer.width, brushLayer.height,
                        0, 0, Math.round(brushLayer.width * scaleX), Math.round(brushLayer.height * scaleY)
                    );
                }

                // 依据当前 slider 的 bbox 裁切 ROI 与画笔层
                const bbox = currentSliderBbox || { x: 0, y: 0, width: roiImg.width, height: roiImg.height };
                const w = Math.max(1, Math.round(bbox.width));
                const h = Math.max(1, Math.round(bbox.height));

                const merged = document.createElement('canvas');
                merged.width = w;
                merged.height = h;
                const mergedCtx = merged.getContext('2d');

                // 绘制底图裁切
                mergedCtx.drawImage(
                    roiImg,
                    Math.round(bbox.x), Math.round(bbox.y), w, h,
                    0, 0, w, h
                );
                // 覆盖画笔裁切
                mergedCtx.drawImage(
                    brushOnRoi,
                    Math.round(bbox.x), Math.round(bbox.y), w, h,
                    0, 0, w, h
                );

                const appliedImage = merged.toDataURL('image/png');

                // 更新本地缓存与界面（不保存到服务器）
                const instanceId = currentInstances[currentInstanceIndex];
                const entry = segmentationData[instanceId] || {};
                entry.segmented_image = appliedImage;
                entry.debug_image = appliedImage;
                entry.processed_roi = appliedImage;
                entry.metadata = entry.metadata || {};
                entry.metadata.segmented_bbox = { x: 0, y: 0, width: w, height: h };
                entry.metadata.roi_shape = [h, w, 3];
                segmentationData[instanceId] = entry;

                processedRoiImage = appliedImage;
                currentSegmentedImage = appliedImage;
                currentSliderBbox = { x: 0, y: 0, width: w, height: h };
                roiShape = { width: w, height: h };

                // 仅替换“切割结果”区域的展示
                document.getElementById('segmented-container').innerHTML = `<img src="${appliedImage}" alt="切割结果">`;
                const currentScale = document.getElementById('image-size-slider').value;
                updateImageSize(currentScale);

                // 关闭手动面板，恢复三图
                document.getElementById('manual-adjust-panel').style.display = 'none';
                document.getElementById('bbox-slider-panel').classList.remove('active');
                deactivateBrushCanvas();
                const imagesGrid = document.querySelector('.images-grid');
                if (imagesGrid) imagesGrid.style.display = '';

                // 刷新预览
                updateBboxPreview();
                document.getElementById('instance-info').innerHTML = '<span class="status-badge success">已应用（未保存）</span>';

            } catch (e) {
                console.error('应用失败:', e);
                alert('应用失败: ' + e.message);
            }
        }

        async function applyCustomParams() {
            const customParams = paramsManager.buildCustomParams();

            // 重新请求切割
            const instanceId = currentInstances[currentInstanceIndex];
            document.getElementById('instance-info').innerHTML = '<span class="status-badge loading">重新切割中...</span>';

            try {
                // debug log removed
                const response = await fetch(`${API_BASE}/segment_instances`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        instance_id: instanceId,
                        custom_params: customParams
                    })
                });

                const data = await response.json();
                if (!data.success) {
                    throw new Error(data.error);
                }

                segmentationData[instanceId] = data;

                if (data.metadata && data.metadata.segmented_bbox) {
                    currentSliderBbox = { ...data.metadata.segmented_bbox };
                }
                if (data.metadata && data.metadata.original_bbox) {
                    currentOriginalBbox = { ...data.metadata.original_bbox };
                }
                if (data.metadata && data.metadata.roi_shape) {
                    roiShape = {
                        width: data.metadata.roi_shape[1],
                        height: data.metadata.roi_shape[0]
                    };
                }

                if (data.roi_image) {
                    document.getElementById('roi-container').innerHTML = `<img src="${data.roi_image}" alt="原始ROI">`;
                }

                processedRoiImage = data.processed_roi || null;
                currentSegmentedImage = processedRoiImage || data.segmented_image;

                document.getElementById('segmented-container').innerHTML = `<img src="${data.segmented_image}" alt="切割结果">`;
                document.getElementById('debug-container').innerHTML = `<img src="${data.debug_image}" alt="调试图">`;

                const currentScale = document.getElementById('image-size-slider').value;
                updateImageSize(currentScale);
                document.getElementById('instance-info').innerHTML = '<span class="status-badge success">切割完成</span>';

                const manualPanel = document.getElementById('manual-adjust-panel');
                if (manualPanel && manualPanel.style.display !== 'none') {
                    initializeBboxSlider();
                    updateBboxPreview();
                }

            } catch (error) {
                console.error('重新切割失败:', error);
                document.getElementById('instance-info').innerHTML = `<span class="status-badge error">切割失败: ${error.message}</span>`;
            }
        }

        async function resetParams() {
            // 保存当前滚动位置
            const sidebar = document.querySelector('.sidebar-content');
            const scrollTop = sidebar ? sidebar.scrollTop : 0;

            // 重新生成参数面板，这会重置所有滑块和开关到默认值
            await paramsManager.rebuildPanel();

            // 恢复滚动位置
            if (sidebar) {
                // 使用 requestAnimationFrame 确保 DOM 更新完成
                requestAnimationFrame(() => {
                    sidebar.scrollTop = scrollTop;
                });
            }

            // 重新请求切割，不传 custom_params，让后端使用完整的默认配置
            const instanceId = currentInstances[currentInstanceIndex];
            document.getElementById('instance-info').innerHTML = '<span class="status-badge loading">重新切割中...</span>';

            try {
                // debug log removed
                const response = await fetch(`${API_BASE}/segment_instances`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        instance_id: instanceId
                        // 注意：不传 custom_params，让后端使用完整的默认配置
                    })
                });

                const data = await response.json();
                if (!data.success) {
                    throw new Error(data.error);
                }

                // 更新显示
                segmentationData[instanceId] = data;
                document.getElementById('segmented-container').innerHTML = `<img src="${data.segmented_image}" alt="切割结果">`;
                document.getElementById('debug-container').innerHTML = `<img src="${data.debug_image}" alt="调试图">`;

                // 重新应用当前的图片缩放设置
                const currentScale = document.getElementById('image-size-slider').value;
                updateImageSize(currentScale);
                document.getElementById('instance-info').innerHTML = '<span class="status-badge success">切割完成</span>';

            } catch (error) {
                console.error('重新切割失败:', error);
                document.getElementById('instance-info').innerHTML = `<span class="status-badge error">切割失败: ${error.message}</span>`;
            }
        }

        async function closeModal() {
            document.getElementById('review-modal').classList.remove('active');
            // 解锁 body 滚动
            document.body.style.overflow = '';
            // 关闭所有面板
            document.getElementById('bbox-slider-panel').classList.remove('active');
            const manualPanel = document.getElementById('manual-adjust-panel');
            if (manualPanel) manualPanel.style.display = 'none';
            deactivateBrushCanvas();

            const imagesGrid = document.querySelector('.images-grid');
            if (imagesGrid) imagesGrid.style.display = '';

            // 重置图片大小
            ['roi-container', 'segmented-container', 'debug-container'].forEach(id => {
                const container = document.getElementById(id);
                if (container) {
                    container.style.transform = 'scale(1)';
                    container.classList.remove('scaled');
                }
            });

            // 重置图片大小控制（默认 300%）
            document.getElementById('image-size-slider').value = 300;
            document.getElementById('image-size-value').textContent = '300';

            // 重置 bbox 预览图片大小
            [document.getElementById('bbox-base-canvas'), document.getElementById('bbox-overlay-canvas'), document.getElementById('brush-canvas')].forEach(canvasEl => {
                if (canvasEl) {
                    canvasEl.style.transform = 'scale(1)';
                }
            });

            // 保存当前字符，因为 loadBook 会重新生成 DOM
            const savedChar = currentChar;

            // 刷新书籍列表以更新统计
            if (currentBook) {
                // 刷新统计但保留已访问集合，保证自动跳过重复字
                await loadBook(currentBook, { preserveVisited: true });

                // loadBook 完成后，重新找到当前字符元素并滚动
                if (savedChar) {
                    // 使用 requestAnimationFrame 确保 DOM 已更新
                    requestAnimationFrame(() => {
                        // 查找包含当前字符的元素
                        const charItems = document.querySelectorAll('.char-item');
                        for (const item of charItems) {
                            const charText = item.querySelector('.char-text');
                            if (charText && charText.textContent === savedChar) {
                                item.scrollIntoView({ behavior: 'auto', block: 'center' });
                                break;
                            }
                        }
                    });
                }
            }

            if (pendingAutoIndex != null) {
                requestAnimationFrame(() => {
                    openCharByIndex(pendingAutoIndex);
                    pendingAutoIndex = null;
                });
            } else if (pendingAutoChar) {
                // 兼容旧逻辑（不推荐）：按名称会选中第一个出现位置
                requestAnimationFrame(() => {
                    openCharByName(pendingAutoChar);
                    pendingAutoChar = null;
                });
            }
        }

        // ==================== 手动调整功能（整合 Bbox + 画笔）====================

        function toggleManualAdjustPanel() {
            const panel = document.getElementById('manual-adjust-panel');
            const bboxSliderPanel = document.getElementById('bbox-slider-panel');

            if (panel.style.display === 'none') {
              // 打开手动调整面板
              panel.style.display = 'block';

                  // 自动打开 Bbox Slider（用于显示预览图）
                  bboxSliderPanel.classList.add('active');
                  initializeBboxSlider();

                  // 初始化画笔（在预览画布上）
                  setTimeout(() => initBrushCanvasOnBbox(), 100);

                  // 进入手动调整时，隐藏原/切割/调试三图，只显示正在处理的画布
                  const imagesGrid = document.querySelector('.images-grid');
                  if (imagesGrid) imagesGrid.style.display = 'none';
              } else {
                // 关闭手动调整面板
                  panel.style.display = 'none';
                  bboxSliderPanel.classList.remove('active');
                  deactivateBrushCanvas();

                  // 退出手动调整时，恢复三图显示
                  const imagesGrid = document.querySelector('.images-grid');
                  if (imagesGrid) imagesGrid.style.display = '';
              }
        }

        function cancelManualAdjustment() {
            cancelBboxSliderAdjustment(true);
            document.getElementById('manual-adjust-panel').style.display = 'none';
            document.getElementById('bbox-slider-panel').classList.remove('active');
            deactivateBrushCanvas();
            const imagesGrid = document.querySelector('.images-grid');
            if (imagesGrid) imagesGrid.style.display = '';
        }

        function initBrushCanvasOnBbox() {
            const baseCanvas = document.getElementById('bbox-base-canvas');
            brushCanvas = document.getElementById('brush-canvas');

            if (!baseCanvas || !brushCanvas) {
                console.error('[画笔] 找不到预览画布');
                return;
            }

            brushCanvas.width = baseCanvas.width;
            brushCanvas.height = baseCanvas.height;
            brushCanvas.style.width = baseCanvas.style.width || `${baseCanvas.width}px`;
            brushCanvas.style.height = baseCanvas.style.height || `${baseCanvas.height}px`;

            // 使用 willReadFrequently 提升 getImageData 的读回性能
            brushCtx = brushCanvas.getContext('2d', { willReadFrequently: true });
            // 确保可以接收指针事件
            brushCanvas.style.pointerEvents = 'auto';
            brushCanvas.style.touchAction = 'none';
            // 轻微底色便于确认画布层可点（极淡，不影响观感）
            brushCanvas.style.backgroundColor = 'rgba(0,0,0,0.01)';
            if (brushCtx) {
                brushCtx.clearRect(0, 0, brushCanvas.width, brushCanvas.height);
            }

            // 保存当前显示的图片（用于后续合并）
            if (processedRoiImage) {
                currentSegmentedImage = processedRoiImage;
            } else {
                const segmentedContainer = document.getElementById('segmented-container');
                const segmentedImg = segmentedContainer ? segmentedContainer.querySelector('img') : null;
                if (segmentedImg) {
                    currentSegmentedImage = segmentedImg.src;
                }
            }

            // 解除旧的事件监听器（如果有）
            brushCanvas.removeEventListener('mousedown', startDrawing);
            brushCanvas.removeEventListener('mousemove', draw);
            brushCanvas.removeEventListener('mouseup', stopDrawing);
            brushCanvas.removeEventListener('mouseout', stopDrawing);

            // 绑定新的事件（鼠标 + 触摸 + 指针）
            brushCanvas.addEventListener('mousedown', startDrawing, { passive: false });
            brushCanvas.addEventListener('mousemove', draw, { passive: false });
            brushCanvas.addEventListener('mouseup', stopDrawing, { passive: false });
            brushCanvas.addEventListener('mouseleave', stopDrawing, { passive: false });
            // pointer 事件（兼容性更好）
            brushCanvas.addEventListener('pointerdown', startDrawing, { passive: false });
            brushCanvas.addEventListener('pointermove', draw, { passive: false });
            brushCanvas.addEventListener('pointerup', stopDrawing, { passive: false });
            brushCanvas.addEventListener('pointerleave', stopDrawing, { passive: false });
            
            brushCanvas.addEventListener('touchstart', (e) => {
                if (!brushCanvas) return;
                const t = e.touches[0];
                if (!t) return;
                const rect = brushCanvas.getBoundingClientRect();
                const scaleX = brushCanvas.width / rect.width;
                const scaleY = brushCanvas.height / rect.height;
                const x = (t.clientX - rect.left) * scaleX;
                const y = (t.clientY - rect.top) * scaleY;
                isDrawing = true;
                brushCtx.fillStyle = brushColor;
            brushCtx.beginPath();
            const touchRadius = Math.max(0.5, (brushSize * roiToPreviewScale) / 2);
            brushCtx.arc(x, y, touchRadius, 0, Math.PI * 2);
            brushCtx.fill();
                brushCtx.beginPath();
                brushCtx.moveTo(x, y);
                e.preventDefault();
            }, { passive: false });
            brushCanvas.addEventListener('touchmove', (e) => {
                if (!isDrawing || !brushCanvas) return;
                const t = e.touches[0];
                if (!t) return;
                const rect = brushCanvas.getBoundingClientRect();
                const scaleX = brushCanvas.width / rect.width;
                const scaleY = brushCanvas.height / rect.height;
                const x = (t.clientX - rect.left) * scaleX;
                const y = (t.clientY - rect.top) * scaleY;
                brushCtx.strokeStyle = brushColor;
                const lw = Math.max(1, Math.round(brushSize * roiToPreviewScale));
                brushCtx.lineWidth = lw;
                brushCtx.lineCap = 'round';
                brushCtx.lineJoin = 'round';
                brushCtx.lineTo(x, y);
                brushCtx.stroke();
                e.preventDefault();
            }, { passive: false });
            brushCanvas.addEventListener('touchend', (e) => { stopDrawing(); e.preventDefault(); }, { passive: false });

            // 改变光标样式
            brushCanvas.style.cursor = 'crosshair';

            // 初始化历史记录
            brushHistory = [];
            brushHistoryStep = -1;
            saveHistory();

            
        }

        function deactivateBrushCanvas() {
            // 移除事件监听器，但不删除画布
            if (brushCanvas) {
                brushCanvas.removeEventListener('mousedown', startDrawing);
                brushCanvas.removeEventListener('mousemove', draw);
                brushCanvas.removeEventListener('mouseup', stopDrawing);
                brushCanvas.removeEventListener('mouseout', stopDrawing);
                brushCanvas.style.cursor = '';
                brushCanvas = null;
                brushCtx = null;
            }
            brushHistory = [];
            brushHistoryStep = -1;
        }

        function startDrawing(e) {
            if (!brushCanvas || !brushCtx) return;
            isDrawing = true;
            const rect = brushCanvas.getBoundingClientRect();
            const scaleX = brushCanvas.width / rect.width;
            const scaleY = brushCanvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            // 先落一个点，避免只点击不移动时无反馈
            brushCtx.fillStyle = brushColor;
            brushCtx.beginPath();
            const initialRadius = Math.max(0.5, (brushSize * roiToPreviewScale) / 2);
            brushCtx.arc(x, y, initialRadius, 0, Math.PI * 2);
            brushCtx.fill();

            // 准备连续绘制
            brushCtx.beginPath();
            brushCtx.moveTo(x, y);
            
        }

        function draw(e) {
            if (!isDrawing || !brushCanvas || !brushCtx) return;

            const rect = brushCanvas.getBoundingClientRect();
            const scaleX = brushCanvas.width / rect.width;
            const scaleY = brushCanvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            brushCtx.strokeStyle = brushColor;
            const lw = Math.max(1, Math.round(brushSize * roiToPreviewScale));
            brushCtx.lineWidth = lw;
            brushCtx.lineCap = 'round';
            brushCtx.lineJoin = 'round';
            brushCtx.lineTo(x, y);
            brushCtx.stroke();
            
        }

        function stopDrawing() {
            if (!brushCanvas || !brushCtx) return;
            if (isDrawing) {
                isDrawing = false;
                try { saveHistory(); } catch (e) {}
            }
        }

        function setBrushColor(color) {
            brushColor = color;

            // 更新按钮状态
            document.getElementById('brush-black').classList.toggle('active', color === 'black');
            document.getElementById('brush-white').classList.toggle('active', color === 'white');
        }

          function setBrushSize(size) {
              brushSize = parseInt(size, 10) || 1;
              const v = document.getElementById('brush-size-value');
              if (v) v.textContent = brushSize;
          }

        function saveHistory() {
            // 删除当前步骤之后的历史
            brushHistory = brushHistory.slice(0, brushHistoryStep + 1);

            // 保存当前 Canvas 状态
            const imageData = brushCtx.getImageData(0, 0, brushCanvas.width, brushCanvas.height);
            brushHistory.push(imageData);
            brushHistoryStep++;

            // 限制历史记录数量（最多 20 步）
            if (brushHistory.length > 20) {
                brushHistory.shift();
                brushHistoryStep--;
            }
        }

        function undoBrush() {
            if (brushHistoryStep > 0) {
                brushHistoryStep--;
                const imageData = brushHistory[brushHistoryStep];
                brushCtx.putImageData(imageData, 0, 0);
            }
        }

        function redoBrush() {
            if (brushHistoryStep < brushHistory.length - 1) {
                brushHistoryStep++;
                const imageData = brushHistory[brushHistoryStep];
                brushCtx.putImageData(imageData, 0, 0);
            }
        }

        function clearBrush() {
            if (!confirm('确认清空所有画笔编辑？')) {
                return;
            }

            brushCtx.clearRect(0, 0, brushCanvas.width, brushCanvas.height);
            saveHistory();
        }

        async function saveManualAdjustment() {
            const baseCanvas = document.getElementById('bbox-base-canvas');
            const brushLayer = document.getElementById('brush-canvas');

            if (!baseCanvas) {
                alert('没有可保存的调整');
                return;
            }

            if (!confirm('确认保存手动调整？\n这将保存 Bbox 调整和画笔修改。')) {
                return;
            }

            try {
                const mergeCanvas = document.createElement('canvas');
                mergeCanvas.width = baseCanvas.width;
                mergeCanvas.height = baseCanvas.height;
                const mergeCtx = mergeCanvas.getContext('2d');

                mergeCtx.drawImage(baseCanvas, 0, 0);
                if (brushLayer) {
                    mergeCtx.drawImage(brushLayer, 0, 0);
                }

                const adjustedImage = mergeCanvas.toDataURL('image/png');

                // 保存到服务器
                const instanceId = currentInstances[currentInstanceIndex];
                const response = await fetch(`${API_BASE}/save_segmentation`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        book: currentBook,
                        char: currentChar,
                        instance_id: instanceId,
                        status: 'confirmed',
                        method: 'manual_adjust',
                        segmented_image_base64: adjustedImage,
                        decision: 'need'
                    })
                });

                const result = await response.json();
                if (!result.success) {
                    throw new Error(result.error);
                }

                // 更新状态
                if (!reviewStatus[currentChar]) reviewStatus[currentChar] = {};
                reviewStatus[currentChar][instanceId] = { status: 'confirmed', method: 'manual_adjust', decision: 'need' };

                const segmentationEntry = segmentationData[instanceId] || {};
                segmentationEntry.segmented_image = adjustedImage;
                segmentationEntry.processed_roi = adjustedImage;
                segmentationEntry.debug_image = adjustedImage;
                segmentationEntry.metadata = segmentationEntry.metadata || {};
                segmentationEntry.metadata.segmented_bbox = {
                    x: 0,
                    y: 0,
                    width: baseCanvas.width,
                    height: baseCanvas.height
                };
                segmentationEntry.metadata.roi_shape = [baseCanvas.height, baseCanvas.width, 3];
                segmentationData[instanceId] = segmentationEntry;

                processedRoiImage = adjustedImage;
                currentSegmentedImage = adjustedImage;
                currentSliderBbox = { x: 0, y: 0, width: baseCanvas.width, height: baseCanvas.height };
                roiShape = { width: baseCanvas.width, height: baseCanvas.height };

                document.getElementById('roi-container').innerHTML = `<img src="${adjustedImage}" alt="原始ROI">`;
                document.getElementById('segmented-container').innerHTML = `<img src="${adjustedImage}" alt="切割结果">`;
                document.getElementById('debug-container').innerHTML = `<img src="${adjustedImage}" alt="调试图">`;

                updateBboxPreview();

                alert('保存成功！');

                // 关闭手动调整面板
                document.getElementById('manual-adjust-panel').style.display = 'none';
                document.getElementById('bbox-slider-panel').classList.remove('active');
                deactivateBrushCanvas();
                const imagesGrid = document.querySelector('.images-grid');
                if (imagesGrid) imagesGrid.style.display = '';

                // 刷新显示
                await loadInstance();

            } catch (error) {
                console.error('保存手动调整失败:', error);
                alert('保存失败: ' + error.message);
            }
        }
})();
