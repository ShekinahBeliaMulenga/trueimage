document.addEventListener("DOMContentLoaded", () => {
    
  // 1. Spool up the Progress Bars
  setTimeout(() => {
    const meters = document.querySelectorAll('.meter-bar');
    meters.forEach(meter => {
      meter.style.width = meter.getAttribute('data-width');
    });
  }, 400); // Small delay to sync with CSS fade-in

  // 2. Count Up Numbers
  setTimeout(() => {
    const counters = document.querySelectorAll('.num-counter');
    counters.forEach(counter => {
      const target = parseFloat(counter.getAttribute('data-target'));
      const duration = 1200; // 1.2 seconds to count up
      const start = performance.now();
      
      function updateCounter(currentTime) {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        
        // easeOutExpo for a cool fast-to-slow down effect
        const easeOut = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
        
        // Check if number is an integer (like face count) or float (like probability)
        const isInteger = target % 1 === 0;
        const currentVal = easeOut * target;
        
        counter.textContent = isInteger ? Math.round(currentVal) : currentVal.toFixed(1);
        
        if (progress < 1) {
          requestAnimationFrame(updateCounter);
        } else {
          counter.textContent = target; // Ensure exact final hit
        }
      }
      requestAnimationFrame(updateCounter);
    });
  }, 500);

  // 3. Typewriter Effect for Terminal Message
  setTimeout(() => {
    const terminalBlock = document.getElementById('terminal-block');
    if (!terminalBlock) return; // Safety check
    
    const textTarget = document.getElementById('terminal-text');
    const message = terminalBlock.getAttribute('data-message');
    let index = 0;
    
    function typeChar() {
      if (index < message.length) {
        textTarget.textContent += message.charAt(index);
        index++;
        // Randomize typing speed slightly for realism (5ms - 25ms)
        setTimeout(typeChar, Math.random() * 20 + 5);
      } else {
        // Stop cursor blinking after done typing
        textTarget.classList.remove('cursor');
      }
    }
    typeChar();
  }, 2000); // Wait 2 seconds before starting to type

  // ---------------------------------------------------------
    // 4. Zero Retention Lockdown Timer (Reload-Proof)
    // ---------------------------------------------------------
    const retentionTimeSeconds = 45; 
    
    // Create a unique ID for this specific scan based on the URL
    // This ensures if they scan a NEW image, the timer starts fresh
    const currentScanId = window.location.pathname + window.location.search; 

    // Check if an expiration time is already saved in the browser for this scan
    let expirationTimeMS = sessionStorage.getItem('purge_time_' + currentScanId);

    if (!expirationTimeMS) {
        // First time viewing this result: calculate exact expiration time and save it
        expirationTimeMS = Date.now() + (retentionTimeSeconds * 1000);
        sessionStorage.setItem('purge_time_' + currentScanId, expirationTimeMS);
    }

    // Function to handle the UI lockdown
    function triggerLockdown() {
        const overlay = document.getElementById("purge-overlay");
        if (overlay) {
            overlay.classList.remove("opacity-0", "pointer-events-none");
            overlay.classList.add("opacity-100", "pointer-events-auto");
        }

        const mainContent = document.querySelector("main");
        if (mainContent) {
            mainContent.style.filter = "blur(10px) grayscale(100%)";
            mainContent.style.transition = "filter 1s ease-in-out";
        }
    }

    // Function to calculate remaining time and update UI
    function tickTimer() {
        const now = Date.now();
        const timeLeftMS = expirationTimeMS - now;
        const secondsLeft = Math.ceil(timeLeftMS / 1000);

        // If time is up, lock it down and stop the clock
        if (secondsLeft <= 0) {
            clearInterval(countdownInterval);
            triggerLockdown();
            return;
        }

        // Otherwise, update the header text
        const headerElements = document.querySelectorAll('header span');
        headerElements.forEach(el => {
            if (el.textContent.includes('Sys Online') || el.textContent.includes('Purge In')) {
                el.textContent = `Purge In: ${secondsLeft}s`;
                el.classList.add('text-warning');
            }
        });
    }

    // Immediately check if time expired while they were reloading
    if (Date.now() >= expirationTimeMS) {
        triggerLockdown();
    } else {
        // Run once immediately, then start the 1-second loop
        tickTimer(); 
        var countdownInterval = setInterval(tickTimer, 1000);
    }

});