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
    }, 2000); // Wait 1 second before starting to type
  
  });