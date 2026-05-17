(function (window) {
  const DEFAULT_PARAMS = {
    noise_removal: {
      enabled: true,
      dark_stroke_threshold: 60,
      light_noise_threshold: 240,
      min_stroke_area: 6,
      min_noise_area: 2,
      max_noise_area: 200000,
      noise_threshold: 0.4,
      smart_removal_preserve_distance: 1.0,
      morphology: {
        aspect_ratio: {
          edge_threshold: 5.0,
          noise_threshold: 2.0,
          weight: 0.4,
        },
        solidity: {
          edge_threshold: 0.8,
          noise_threshold: 0.5,
          weight: 0.3,
        },
        perimeter_area: {
          edge_threshold: 50.0,
          noise_threshold: 20.0,
          weight: 0.3,
        },
      },
      distance: {
        mean_distance: {
          edge_threshold: 1.5,
          noise_threshold: 2.0,
          weight: 0.5,
        },
        distance_cv: {
          edge_threshold: 0.2,
          noise_threshold: 0.25,
          weight: 0.5,
        },
      },
      feature_weights: {
        morphology: 0.4,
        distance: 0.6,
      },
      debug_features: false,
      debug_visualize: true,
    },
    projection_trim: {
      enabled: true,
      binarize: 'otsu',
      adaptive_block: 31,
      adaptive_C: 3,
      run_min_coverage_ratio: 0.01,
      run_min_coverage_abs: 0.005,
      primary_run_min_mass_ratio: 0.5,
      primary_run_min_length_ratio: 0.3,
      tighten_min_coverage: 0.01,
      detection_range: {
        left_ratio: 0.3,
        right_ratio: 0.3,
        top_ratio: 0.3,
        bottom_ratio: 0.5,
      },
      cut_limits: {
        left_max_ratio: 0.25,
        right_max_ratio: 0.25,
        top_max_ratio: 0.25,
        bottom_max_ratio: 0.3,
      },
    },
    cc_filter: {
      enabled: true,
      border_touch_margin: 1,
      edge_zone_margin: 2,
      border_touch_min_area_ratio: 0.04,
      edge_zone_min_area_ratio: 0.01,
      interior_min_area_ratio: 0.002,
      max_aspect_for_edge: 6.0,
      min_dim_px: 2,
      interior_min_dim_px: 1,
      debug_visualize: true,
    },
    border_removal: {
      enabled: true,
      max_iterations: 5,
      border_max_width_ratio: 0.15,
      border_threshold_ratio: 0.35,
      spike_min_length_ratio: 0.02,
      spike_max_length_ratio: 0.1,
      spike_gradient_threshold: 0.4,
      spike_prominence_ratio: 0.5,
      edge_tolerance: 2,
      vertical_detection_range: {
        top_ratio: 0.3,
        bottom_ratio: 0.3,
      },
      vertical_cut_limits: {
        top_max_ratio: 0.2,
        bottom_max_ratio: 0.2,
      },
      debug_verbose: true,
    }
  };

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function createSubsectionTitle(text) {
    const element = document.createElement('div');
    element.className = 'param-subsection-title';
    element.textContent = text;
    return element;
  }

  function inferSliderBounds(name) {
    let min = 0;
    let max = 10;
    let step = 0.1;

    if (name.includes('ratio')) {
      max = 1;
      step = 0.01;
    } else if (name.includes('_px') || name.includes('block') || name.includes('threshold') || name.includes('area')) {
      if (name.includes('area')) {
        max = 500;
        step = 1;
      } else if (name.includes('block')) {
        max = 101;
        step = 2;
      } else if (name.includes('threshold')) {
        max = 255;
        step = 1;
      } else {
        max = 50;
        step = 1;
      }
    } else if (name.includes('distance')) {
      max = 10;
      step = 0.1;
    } else if (name.includes('weight')) {
      max = 1;
      step = 0.01;
    }

    return { min, max, step };
  }

  function createParamSlider(name, value, label, description) {
    const item = document.createElement('div');
    item.className = 'param-item';

    const labelRow = document.createElement('div');
    labelRow.className = 'param-label';

    const nameSpan = document.createElement('span');
    nameSpan.textContent = label || name;

    const valueSpan = document.createElement('span');
    valueSpan.className = 'param-value';
    valueSpan.id = `param-value-${name}`;
    valueSpan.textContent = value;

    labelRow.appendChild(nameSpan);
    labelRow.appendChild(valueSpan);
    item.appendChild(labelRow);

    if (description) {
      const descriptionEl = document.createElement('div');
      descriptionEl.className = 'param-description';
      descriptionEl.textContent = description;
      item.appendChild(descriptionEl);
    }

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'param-slider';
    slider.id = `param-${name}`;

    const bounds = inferSliderBounds(name);
    slider.min = bounds.min;
    slider.max = bounds.max;
    slider.step = bounds.step;
    slider.value = value;
    slider.addEventListener('input', event => {
      valueSpan.textContent = event.target.value;
    });

    item.appendChild(slider);
    return item;
  }

  function createBooleanParam(name, value, label) {
    const item = document.createElement('div');
    item.className = 'param-item';

    const labelRow = document.createElement('div');
    labelRow.className = 'param-label';
    const title = document.createElement('span');
    title.textContent = label;
    labelRow.appendChild(title);
    item.appendChild(labelRow);

    const toggleLabel = document.createElement('label');
    toggleLabel.className = 'param-boolean-label';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = `param-${name}`;
    checkbox.checked = !!value;

    const valueSpan = document.createElement('span');
    valueSpan.id = `param-value-${name}`;
    valueSpan.className = 'param-boolean-value';
    valueSpan.textContent = value ? 'true' : 'false';

    checkbox.addEventListener('change', event => {
      valueSpan.textContent = event.target.checked ? 'true' : 'false';
    });

    toggleLabel.appendChild(checkbox);
    toggleLabel.appendChild(valueSpan);
    item.appendChild(toggleLabel);

    return item;
  }

  function appendNumberParams(container, entries, prefix) {
    entries.forEach(entry => {
      const key = Array.isArray(entry) ? entry[0] : entry.key;
      const label = Array.isArray(entry) ? entry[1] : entry.key;
      const description = Array.isArray(entry) ? undefined : entry.desc;
      const value = prefix.source[key];
      if (typeof value === 'number') {
        container.appendChild(createParamSlider(`${prefix.path}.${key}`, value, label, description));
      }
    });
  }

  function createNoiseRemovalParams(params) {
    const container = document.createElement('div');
    appendNumberParams(container, [
      { key: 'dark_stroke_threshold', desc: 'Values ≤ this are considered text strokes' },
      { key: 'light_noise_threshold', desc: 'Areas > dark_stroke and < this are noise candidates' },
      { key: 'min_stroke_area', desc: 'Filter dark spots smaller than this' },
      { key: 'min_noise_area', desc: 'Preserve noise smaller than this' },
      { key: 'max_noise_area', desc: 'Directly classify as noise if larger than this' },
      { key: 'noise_threshold', desc: 'Classify as noise if score > this' },
      { key: 'smart_removal_preserve_distance', desc: 'Pixels < this distance from stroke edge are preserved' }
    ], { source: params, path: 'noise_removal' });
    return container;
  }

  function createCCFilterParams(params) {
    const container = document.createElement('div');
    appendNumberParams(container, [
      ['border_touch_margin', 'border_touch_margin'],
      ['edge_zone_margin', 'edge_zone_margin'],
      ['border_touch_min_area_ratio', 'border_touch_min_area_ratio'],
      ['edge_zone_min_area_ratio', 'edge_zone_min_area_ratio'],
      ['interior_min_area_ratio', 'interior_min_area_ratio'],
      ['max_aspect_for_edge', 'max_aspect_for_edge'],
      ['min_dim_px', 'min_dim_px'],
      ['interior_min_dim_px', 'interior_min_dim_px']
    ], { source: params, path: 'cc_filter' });
    return container;
  }

  function appendNestedSection(container, title, pathPrefix, source) {
    if (!source) return;
    const section = document.createElement('div');
    section.className = 'param-subsection';
    section.appendChild(createSubsectionTitle(title));

    for (const [subKey, subValue] of Object.entries(source)) {
      if (typeof subValue === 'number') {
        section.appendChild(createParamSlider(`${pathPrefix}.${subKey}`, subValue, subKey));
      }
    }

    container.appendChild(section);
  }

  function createProjectionTrimParams(params) {
    const container = document.createElement('div');
    appendNumberParams(container, [
      ['adaptive_block', 'adaptive_block'],
      ['adaptive_C', 'adaptive_C'],
      ['run_min_coverage_ratio', 'run_min_coverage_ratio'],
      ['run_min_coverage_abs', 'run_min_coverage_abs'],
      ['primary_run_min_mass_ratio', 'primary_run_min_mass_ratio'],
      ['primary_run_min_length_ratio', 'primary_run_min_length_ratio'],
      ['tighten_min_coverage', 'tighten_min_coverage']
    ], { source: params, path: 'projection_trim' });

    appendNestedSection(container, '检测范围', 'projection_trim.detection_range', params.detection_range);
    appendNestedSection(container, '切割限制', 'projection_trim.cut_limits', params.cut_limits);
    return container;
  }

  function createBorderRemovalParams(params) {
    const container = document.createElement('div');
    appendNumberParams(container, [
      ['max_iterations', 'max_iterations'],
      ['border_max_width_ratio', 'border_max_width_ratio'],
      ['border_threshold_ratio', 'border_threshold_ratio'],
      ['spike_min_length_ratio', 'spike_min_length_ratio'],
      ['spike_max_length_ratio', 'spike_max_length_ratio'],
      ['spike_gradient_threshold', 'spike_gradient_threshold'],
      ['spike_prominence_ratio', 'spike_prominence_ratio'],
      ['edge_tolerance', 'edge_tolerance']
    ], { source: params, path: 'border_removal' });

    const combined = document.createElement('div');
    combined.className = 'param-subsection';
    combined.appendChild(createSubsectionTitle('垂直边框参数'));
    ['vertical_detection_range', 'vertical_cut_limits'].forEach(group => {
      if (params[group]) {
        for (const [subKey, subValue] of Object.entries(params[group])) {
          if (typeof subValue === 'number') {
            combined.appendChild(createParamSlider(`border_removal.${group}.${subKey}`, subValue, subKey));
          }
        }
      }
    });
    container.appendChild(combined);
    return container;
  }

  function updateNestedValue(target, name, value, logLabel) {
    const parts = name.split('.');
    if (parts.length < 2) return;

    const category = parts[0];
    const path = parts.slice(1);
    if (!target[category]) return;

    let current = target[category];
    for (let i = 0; i < path.length - 1; i += 1) {
      if (current[path[i]] !== undefined) {
        current = current[path[i]];
      } else {
        console.warn(`${logLabel}路径 ${name} 在默认模板中不存在: ${path[i]}`);
        return;
      }
    }

    const finalKey = path[path.length - 1];
    if (current[finalKey] !== undefined) {
      current[finalKey] = value;
    } else {
      console.warn(`${logLabel}${name} 在默认模板中不存在: ${finalKey}`);
    }
  }

  function createManager(options) {
    const { apiBase, contentId = 'params-content' } = options;

    function getContentEl() {
      return document.getElementById(contentId);
    }

    function toggleParameterGroup(toggleId) {
      const isEnabled = document.getElementById(toggleId).checked;
      const group = document.getElementById(`param-group-${toggleId}`);
      if (!group) return;
      group.classList.toggle('disabled', !isEnabled);
    }

    function createParameterGroup(title, toggleId, params, createParamsFunc) {
      const group = document.createElement('div');
      group.className = 'param-group';
      group.id = `param-group-${toggleId}`;

      const header = document.createElement('div');
      header.className = 'param-group-header';

      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.id = toggleId;
      checkbox.checked = true;
      checkbox.addEventListener('change', () => toggleParameterGroup(toggleId));

      const heading = document.createElement('h4');
      heading.textContent = title;

      label.appendChild(checkbox);
      label.appendChild(heading);
      header.appendChild(label);

      const content = document.createElement('div');
      content.className = 'param-group-content';
      content.appendChild(createParamsFunc(params));

      group.appendChild(header);
      group.appendChild(content);
      return group;
    }

    async function initPanel() {
      const content = getContentEl();
      if (!content || content.children.length > 1) return;

      try {
        const response = await fetch(`${apiBase}/get_default_params`);
        const data = await response.json();
        const params = data.params;

        content.appendChild(createParameterGroup('Noise Removal', 'toggle-noise', params.noise_removal, createNoiseRemovalParams));
        content.appendChild(createParameterGroup('CC Filtering', 'toggle-cc', params.cc_filter, createCCFilterParams));
        content.appendChild(createParameterGroup('Projection Trimming', 'toggle-projection', params.projection_trim, createProjectionTrimParams));
        content.appendChild(createParameterGroup('Border Removal', 'toggle-border', params.border_removal, createBorderRemovalParams));
      } catch (error) {
        console.error('加载参数失败:', error);
        const failure = document.createElement('p');
        failure.className = 'param-load-error';
        failure.textContent = '参数加载失败';
        content.appendChild(failure);
      }
    }

    async function rebuildPanel() {
      const content = getContentEl();
      if (!content) return;
      content.innerHTML = '';
      await initPanel();
    }

    function buildCustomParams() {
      const customParams = deepClone(DEFAULT_PARAMS);

      const noiseToggle = document.getElementById('toggle-noise');
      const ccToggle = document.getElementById('toggle-cc');
      const projToggle = document.getElementById('toggle-projection');
      const borderToggle = document.getElementById('toggle-border');

      if (noiseToggle) customParams.noise_removal.enabled = noiseToggle.checked;
      if (ccToggle) customParams.cc_filter.enabled = ccToggle.checked;
      if (projToggle) customParams.projection_trim.enabled = projToggle.checked;
      if (borderToggle) customParams.border_removal.enabled = borderToggle.checked;

      document.querySelectorAll('.param-slider').forEach(slider => {
        updateNestedValue(customParams, slider.id.replace('param-', ''), parseFloat(slider.value), '参数');
      });

      document.querySelectorAll('input[id^="param-"][type="checkbox"]').forEach(checkbox => {
        updateNestedValue(customParams, checkbox.id.replace('param-', ''), checkbox.checked, '布尔参数');
      });

      return customParams;
    }

    return {
      initPanel,
      rebuildPanel,
      buildCustomParams,
    };
  }

  window.SegmentReviewParams = { createManager };
})(window);
