document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const dropInner = document.getElementById("drop-inner");
  const fileInput = document.getElementById("file-input");
  const defaultState = document.getElementById("drop-default");
  const previewState = document.getElementById("drop-preview");
  const previewImg = document.getElementById("preview-img");
  const previewName = document.getElementById("preview-name");
  const previewSize = document.getElementById("preview-size");
  const form = document.getElementById("upload-form");
  const scanLine = document.getElementById("scan-line");
  const statusText = document.getElementById("status-text");

  // Elements for Toggle Feature
  const autoScanToggle = document.getElementById("auto-scan-toggle");
  const toggleLabel = document.getElementById("toggle-label");
  const manualSubmitBtn = document.getElementById("manual-submit-btn");
  const processingModeText = document.getElementById("processing-mode-text");

  const MAX_SIZE = 10 * 1024 * 1024; // 10 MB
  const VALID_TYPES = ["image/jpeg", "image/png"];

  let currentObjectURL = null;
  let autoSubmitTimeout = null;

  if (scanLine) scanLine.classList.add("hidden");
  if (statusText) statusText.classList.add("hidden");

  // Custom Alert Function (Persistent, No Reload Needed)
  function showCustomAlert(message) {
    // Remove any existing alert to prevent stacking
    const existingAlert = document.getElementById("custom-js-alert");
    if (existingAlert) existingAlert.remove();

    // Create the new alert div with matching Tailwind classes
    const alertDiv = document.createElement("div");
    alertDiv.id = "custom-js-alert";
    alertDiv.className = "w-full max-w-3xl mb-6 px-4 py-3 bg-red-950/50 border border-red-500/50 text-red-300 text-sm flex items-center gap-3 backdrop-blur-sm animate-pulse";
    
    alertDiv.innerHTML = `
      <span class="material-symbols-outlined text-red-400">warning</span>
      <span>${message}</span>
    `;

    // Insert it right above the form
    form.parentNode.insertBefore(alertDiv, form);
  }

  // Handle Toggle Switch Changes
  autoScanToggle.addEventListener("change", (e) => {
    const isOn = e.target.checked;
    
    // Update UI text and colors
    toggleLabel.textContent = isOn ? "Auto Scan: ON" : "Auto Scan: OFF";
    toggleLabel.className = isOn 
      ? "text-xs uppercase tracking-[0.15em] text-primary-container transition-colors duration-300" 
      : "text-xs uppercase tracking-[0.15em] text-zinc-500 transition-colors duration-300";

    // Update Processing Mode header
    if (processingModeText) {
      processingModeText.textContent = isOn ? "Auto" : "Manual";
    }

    // Show/Hide Manual Button
    if (isOn) {
      manualSubmitBtn.classList.add("hidden");
      // If a file is already loaded, start the scan immediately
      if (currentObjectURL) {
        statusText.textContent = "Auto scan resumed. Starting scan...";
        autoSubmitTimeout = setTimeout(() => { startAutoScan(); }, 800);
      }
    } else {
      manualSubmitBtn.classList.remove("hidden");
      // Stop any pending auto-scans
      if (autoSubmitTimeout) {
        clearTimeout(autoSubmitTimeout);
        autoSubmitTimeout = null;
      }
      if (currentObjectURL) {
        statusText.textContent = "Preview loaded. Click Scan to begin.";
      }
    }
  });

  // Handle Manual Submit Click
  manualSubmitBtn.addEventListener("click", (e) => {
    e.preventDefault(); // Prevent accidental native form submission
    manualSubmitBtn.disabled = true;
    startAutoScan();
  });

  // Drag and Drop Event Listeners
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => {
      dropInner.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => {
      dropInner.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    handleFiles(e.dataTransfer.files);
  });

  dropInner.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    handleFiles(e.target.files);
  });

  previewImg.addEventListener("click", () => fileInput.click());

  function handleFiles(files) {
    // Clear any existing alerts when a new file is selected
    const existingAlert = document.getElementById("custom-js-alert");
    if (existingAlert) existingAlert.remove();

    if (!files || files.length === 0) {
      resetUpload();
      return;
    }

    const file = files[0];

    // Validate Format
    if (!VALID_TYPES.includes(file.type)) {
      showCustomAlert("Unsupported format. Please select a JPG or PNG image.");
      resetUpload();
      return;
    }

    // Validate Size
    if (file.size > MAX_SIZE) {
      showCustomAlert("File is too large. Maximum size is 10 MB.");
      resetUpload();
      return;
    }

    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
    } catch (err) {
      console.warn("DataTransfer not supported, using fallback.");
    }

    if (currentObjectURL) {
      URL.revokeObjectURL(currentObjectURL);
    }

    if (autoSubmitTimeout) {
      clearTimeout(autoSubmitTimeout);
    }

    currentObjectURL = URL.createObjectURL(file);
    previewImg.src = currentObjectURL;
    previewName.textContent = file.name;
    previewSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";

    defaultState.classList.add("hidden");
    previewState.classList.remove("hidden");

    if (statusText) statusText.classList.remove("hidden");
    
    // Check Toggle State to determine next action
    if (autoScanToggle.checked) {
      statusText.textContent = "Preview loaded. Starting scan...";
      autoSubmitTimeout = setTimeout(() => {
        startAutoScan();
      }, 800);
    } else {
      statusText.textContent = "Preview loaded. Click Scan to begin.";
      manualSubmitBtn.disabled = false;
    }
  }

  // ----------------------------------------------------------------------
  // THE NEW AJAX / FETCH PIPELINE
  // ----------------------------------------------------------------------
  function startAutoScan() {
    if (!fileInput.files || fileInput.files.length === 0) {
      return;
    }

    // UI Updates: Show scanning state
    if (statusText) statusText.textContent = "Analyzing image...";
    if (scanLine) scanLine.classList.remove("hidden");
    manualSubmitBtn.disabled = true; // Lock button during scan

    // 1. Package the file for the background request
    const formData = new FormData(form);

    // 2. Add a slight visual delay for the "Forensic Analysis" feel
    setTimeout(() => {
      fetch("/predict", {
        method: "POST",
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        if (data.status === "error") {
          // Analysis failed (Safety filter tripped or no face detected)
          if (scanLine) scanLine.classList.add("hidden");
          if (statusText) statusText.textContent = "Analysis failed.";
          
          showCustomAlert(data.message);
          
          // Re-enable manual scan if applicable so they can try a new image
          if (!autoScanToggle.checked) {
            manualSubmitBtn.disabled = false;
          }
        } 
        else if (data.status === "success") {
          // Success! Redirect the browser to the result report URL provided by Python
          if (statusText) statusText.textContent = "Analysis complete. Generating report...";
          window.location.href = data.redirect_url;
        }
      })
      .catch(error => {
        // Handle server crashes or network drops
        console.error("Network Error:", error);
        if (scanLine) scanLine.classList.add("hidden");
        if (statusText) statusText.textContent = "Connection error.";
        showCustomAlert("System Error: Could not connect to the forensic analysis server.");
        if (!autoScanToggle.checked) manualSubmitBtn.disabled = false;
      });
    }, 700); // 700ms UI delay
  }

  function resetUpload() {
    if (currentObjectURL) {
      URL.revokeObjectURL(currentObjectURL);
      currentObjectURL = null;
    }

    if (autoSubmitTimeout) {
      clearTimeout(autoSubmitTimeout);
      autoSubmitTimeout = null;
    }

    fileInput.value = "";
    previewImg.src = "";
    previewName.textContent = "";
    previewSize.textContent = "";

    defaultState.classList.remove("hidden");
    previewState.classList.add("hidden");

    if (scanLine) {
      scanLine.classList.add("hidden");
    }

    if (statusText) {
      statusText.textContent = "";
      statusText.classList.add("hidden");
    }
    
    manualSubmitBtn.disabled = true;
  }
});