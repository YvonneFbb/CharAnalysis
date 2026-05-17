(function (window) {
  function qs(selector, root) {
    return (root || document).querySelector(selector);
  }

  function clear(node) {
    if (node) node.replaceChildren();
  }

  function text(node, value) {
    if (node) node.textContent = value == null ? '' : String(value);
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === undefined || value === null) continue;
      if (key === 'className') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'dataset') Object.assign(node.dataset, value);
      else if (key.startsWith('on') && typeof value === 'function') {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else {
        node.setAttribute(key, value);
      }
    }
    for (const child of children || []) {
      if (child !== undefined && child !== null) node.appendChild(child);
    }
    return node;
  }

  function option(value, label) {
    const node = document.createElement('option');
    node.value = value;
    node.textContent = label;
    return node;
  }

  window.ReviewDom = { qs, clear, text, el, option };
})(window);
