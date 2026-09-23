  const classInput = document.getElementById('filterClass');
  const resultSelect = document.getElementById('filterResult');
  const confidenceInput = document.getElementById('minConfidence');
  const sortSelect = document.getElementById('sortBy');
  const tbody = document.querySelector('#reportTable tbody');
  const rows = Array.from(document.querySelectorAll('#reportTable tbody tr'));
  const modelBtn = document.getElementById('modelDropdownBtn');
  const modelPanel = document.getElementById('modelDropdownPanel');
  const modelChecks = Array.from(document.querySelectorAll('.model-checkbox'));

  function selectedModels() {
    return modelChecks.filter((c) => c.checked).map((c) => c.value);
  }

  function updateModelBtnLabel() {
    const selected = modelChecks.filter((c) => c.checked);
    let label;
    if (selected.length === 0) {
      label = 'No models';
    } else if (selected.length === modelChecks.length) {
      label = 'All models';
    } else {
      label = selected.length + ' model' + (selected.length > 1 ? 's' : '');
    }
    modelBtn.textContent = label + ' ▾';
  }

  modelBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    modelPanel.classList.toggle('open');
  });
  document.addEventListener('click', () => modelPanel.classList.remove('open'));
  modelPanel.addEventListener('click', (e) => e.stopPropagation());
  document.getElementById('modelSelectAll').addEventListener('click', () => {
    modelChecks.forEach((c) => { c.checked = true; });
    updateModelBtnLabel();
    applyFilters();
  });
  document.getElementById('modelSelectNone').addEventListener('click', () => {
    modelChecks.forEach((c) => { c.checked = false; });
    updateModelBtnLabel();
    applyFilters();
  });
  modelChecks.forEach((c) => c.addEventListener('change', () => {
    updateModelBtnLabel();
    applyFilters();
  }));

  function rowTop1(row) {
    try { return JSON.parse(row.dataset.top1 || '{}'); } catch (e) { return {}; }
  }

  function rowAgreement(row, models) {
    // True/False only when every selected model has a top-1 result; otherwise
    // indeterminate, rather than silently agreeing/disagreeing over a subset.
    // Identity is the class id: distinct classes can share a display label.
    if (models.length < 2) return null;
    const top1 = rowTop1(row);
    const ids = models.map((m) => top1[m] && top1[m].class_id);
    if (ids.some((id) => id === undefined)) return null;
    return ids.every((id) => id === ids[0]);
  }

  function rowMaxConfidence(row, models) {
    const top1 = rowTop1(row);
    const probs = models.map((m) => top1[m] && top1[m].prob).filter((v) => v !== undefined);
    return probs.length ? Math.max(...probs) : null;
  }

  function rowClassText(row, models) {
    const top1 = rowTop1(row);
    return models.map((m) => (top1[m] && top1[m].label) || '').join(' ');
  }

  function applyFilters() {
    const models = selectedModels();
    const classQuery = classInput.value.trim().toLowerCase();
    const resultQuery = resultSelect.value;
    const minConfidence = confidenceInput.value === '' ? null : parseFloat(confidenceInput.value) / 100;

    document.querySelectorAll('[data-model-col]').forEach((cell) => {
      cell.style.display = models.includes(cell.dataset.modelCol) ? '' : 'none';
    });

    for (const row of rows) {
      const hasError = row.dataset.hasError === '1';
      let visible;

      if (resultQuery === 'error') {
        visible = hasError;
      } else {
        const agree = rowAgreement(row, models);
        if (resultQuery === 'agree') visible = agree === true;
        else if (resultQuery === 'disagree') visible = agree === false;
        else visible = true;
      }

      if (visible && classQuery) {
        visible = rowClassText(row, models).toLowerCase().includes(classQuery);
      }

      if (visible && minConfidence !== null) {
        const maxConf = rowMaxConfidence(row, models);
        visible = maxConf !== null && maxConf >= minConfidence;
      }

      const agreeCell = row.querySelector('.agree-cell');
      if (agreeCell) {
        const agree = rowAgreement(row, models);
        agreeCell.textContent = agree === null ? '—' : (agree ? 'agree' : 'disagree');
      }

      row.style.display = visible ? '' : 'none';
    }

    applySort(models);
  }

  function applySort(models) {
    const sortKey = sortSelect.value;
    const sorted = rows.slice();
    if (sortKey === 'confidence') {
      sorted.sort((a, b) => {
        const av = rowMaxConfidence(a, models);
        const bv = rowMaxConfidence(b, models);
        return (bv === null ? -1 : bv) - (av === null ? -1 : av);
      });
    } else if (sortKey === 'class') {
      sorted.sort((a, b) => rowClassText(a, models).localeCompare(rowClassText(b, models)));
    } else if (sortKey === 'result') {
      sorted.sort((a, b) => {
        const ra = a.dataset.hasError === '1' ? 2 : (rowAgreement(a, models) === false ? 1 : 0);
        const rb = b.dataset.hasError === '1' ? 2 : (rowAgreement(b, models) === false ? 1 : 0);
        return ra - rb;
      });
    } else {
      sorted.sort((a, b) => Number(a.dataset.idx) - Number(b.dataset.idx));
    }
    for (const row of sorted) tbody.appendChild(row);
  }

  updateModelBtnLabel();
  classInput.addEventListener('input', applyFilters);
  resultSelect.addEventListener('change', applyFilters);
  confidenceInput.addEventListener('input', applyFilters);
  sortSelect.addEventListener('change', applyFilters);
  applyFilters();
