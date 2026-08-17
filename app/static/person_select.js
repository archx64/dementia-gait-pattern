(function () {
  const img = document.getElementById("preview-img");
  const clickX = document.getElementById("click_x");
  const clickY = document.getElementById("click_y");
  const confirmBtn = document.getElementById("confirm-btn");
  const wrap = img.parentElement;
  let marker = null;

  img.addEventListener("click", function (evt) {
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;

    // Coordinates in the *original* image resolution -- bboxes from the
    // pose model are in that space, not the browser-rendered display size.
    const x = Math.round((evt.clientX - rect.left) * scaleX);
    const y = Math.round((evt.clientY - rect.top) * scaleY);

    clickX.value = x;
    clickY.value = y;
    confirmBtn.disabled = false;

    if (marker) marker.remove();
    marker = document.createElement("div");
    marker.className = "click-marker";
    marker.style.left = (evt.clientX - rect.left) + "px";
    marker.style.top = (evt.clientY - rect.top) + "px";
    wrap.appendChild(marker);
  });
})();
