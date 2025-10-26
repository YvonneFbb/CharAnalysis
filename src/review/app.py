"""
审查系统后端服务器

提供API：
1. 获取书籍列表和字符数据
2. 按需裁切字符实例图片
3. 返回裁切后的图片文件
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
MATCHED_JSON_PATH = PROJECT_ROOT / 'data/results/matched_by_book.json'
STANDARD_CHARS_JSON_PATH = PROJECT_ROOT / 'data/standard_chars.json'
PREPROCESSED_DIR = PROJECT_ROOT / 'data/results/preprocessed'
CROPPED_CACHE_DIR = PROJECT_ROOT / 'data/results/cropped_cache'

# 创建缓存目录
CROPPED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
matched_data = None
standard_chars_data = None


def load_data():
    """加载匹配数据和标准字数据"""
    global matched_data, standard_chars_data

    print(f"加载匹配数据：{MATCHED_JSON_PATH}")
    with open(MATCHED_JSON_PATH, 'r', encoding='utf-8') as f:
        matched_data = json.load(f)

    print(f"加载标准字数据：{STANDARD_CHARS_JSON_PATH}")
    with open(STANDARD_CHARS_JSON_PATH, 'r', encoding='utf-8') as f:
        standard_chars_data = json.load(f)

    print(f"数据加载完成：{len(matched_data['books'])} 本书，{standard_chars_data['total_methods']} 个分类法")


@app.route('/api/books', methods=['GET'])
def get_books():
    """获取所有书籍列表"""
    books_list = [
        {
            'name': name,
            'total_chars': data['total_standard_chars'],
            'total_instances': data['total_instances']
        }
        for name, data in matched_data['books'].items()
    ]
    return jsonify({
        'success': True,
        'books': sorted(books_list, key=lambda x: x['name'])
    })


@app.route('/api/methods', methods=['GET'])
def get_methods():
    """获取84法分类"""
    return jsonify({
        'success': True,
        'methods': standard_chars_data['methods']
    })


@app.route('/api/book/<book_name>', methods=['GET'])
def get_book_chars(book_name):
    """获取指定书籍的字符列表（按84法分类）"""
    if book_name not in matched_data['books']:
        return jsonify({'success': False, 'error': '书籍不存在'}), 404

    book_data = matched_data['books'][book_name]
    book_chars = book_data['chars']

    # 按84法分类组织字符
    methods_with_chars = []
    for method in standard_chars_data['methods']:
        method_chars = []
        for char in method['chars']:
            if char in book_chars:
                method_chars.append({
                    'char': char,
                    'count': len(book_chars[char])
                })

        if method_chars:  # 只包含有字符的分类
            methods_with_chars.append({
                'id': method['id'],
                'name': method['name'],
                'description': method.get('description', ''),
                'chars': method_chars
            })

    return jsonify({
        'success': True,
        'book_name': book_name,
        'total_chars': book_data['total_standard_chars'],
        'total_instances': book_data['total_instances'],
        'methods': methods_with_chars
    })


@app.route('/api/instances', methods=['GET'])
def get_instances():
    """获取字符的实例列表（支持分页）"""
    book_name = request.args.get('book')
    char = request.args.get('char')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 20))

    if not book_name or not char:
        return jsonify({'success': False, 'error': '缺少参数'}), 400

    if book_name not in matched_data['books']:
        return jsonify({'success': False, 'error': '书籍不存在'}), 404

    book_data = matched_data['books'][book_name]
    if char not in book_data['chars']:
        return jsonify({'success': False, 'error': '字符不存在'}), 404

    instances = book_data['chars'][char]
    total = len(instances)

    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    page_instances = instances[start:end]

    return jsonify({
        'success': True,
        'book_name': book_name,
        'char': char,
        'total': total,
        'page': page,
        'page_size': page_size,
        'total_pages': (total + page_size - 1) // page_size,
        'instances': page_instances
    })


@app.route('/api/crop', methods=['POST'])
def crop_image():
    """裁切单个字符实例"""
    data = request.json
    book_name = data.get('book')
    char = data.get('char')
    instance_index = data.get('index')
    padding = data.get('padding', 5)

    if not all([book_name, char, instance_index is not None]):
        return jsonify({'success': False, 'error': '缺少参数'}), 400

    # 获取实例信息
    try:
        instance = matched_data['books'][book_name]['chars'][char][instance_index]
    except (KeyError, IndexError):
        return jsonify({'success': False, 'error': f'实例不存在: book={book_name}, char={char}, index={instance_index}'}), 404

    # 使用实例数据中的 source_image 路径（更可靠）
    if 'source_image' in instance:
        source_image_path = PROJECT_ROOT / instance['source_image']
    else:
        # 备用方案：手动构建路径
        volume = instance['volume']
        page = instance['page']
        source_image_path = PREPROCESSED_DIR / book_name / f'册{volume:02d}_pages' / f'{page}_preprocessed.png'

    if not source_image_path.exists():
        return jsonify({'success': False, 'error': f'源图片不存在: {source_image_path}'}), 404

    # 裁切图片
    try:
        img = cv2.imread(str(source_image_path))
        if img is None:
            return jsonify({'success': False, 'error': '无法读取图片'}), 500

        bbox = instance['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        # 添加padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        cropped = img[y1:y2, x1:x2]

        # 转换为base64
        _, buffer = cv2.imencode('.png', cropped)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'size': {'width': cropped.shape[1], 'height': cropped.shape[0]}
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/')
def index():
    """主页"""
    return '''
    <h1>古籍字形审查系统 API</h1>
    <p>后端服务运行中...</p>
    <ul>
        <li>GET /api/books - 获取书籍列表</li>
        <li>GET /api/methods - 获取84法分类</li>
        <li>GET /api/book/&lt;book_name&gt; - 获取书籍字符</li>
        <li>GET /api/instances?book=&lt;book&gt;&char=&lt;char&gt;&page=&lt;page&gt; - 获取实例列表</li>
        <li>POST /api/crop - 裁切字符图片</li>
    </ul>
    <p>前端界面请访问：<a href="/review">审查界面</a></p>
    '''


@app.route('/review')
def review_page():
    """审查界面"""
    # 返回 HTML 文件
    html_path = PROJECT_ROOT / 'data/results/review_app.html'
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return '审查界面未生成，请先运行生成脚本', 404


def main():
    """启动服务器"""
    load_data()

    print("\n" + "=" * 60)
    print("古籍字形审查系统 - 后端服务器")
    print("=" * 60)
    print(f"数据文件：{MATCHED_JSON_PATH}")
    print(f"标准字文件：{STANDARD_CHARS_JSON_PATH}")
    print(f"图片目录：{PREPROCESSED_DIR}")
    print("\n服务器启动中...")
    print("=" * 60)
    print("\n访问地址：")
    print("  主页：http://localhost:5001")
    print("  审查界面：http://localhost:5001/review")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
