document.addEventListener("DOMContentLoaded", function() {
    const button = document.querySelector('.btn-custom');
    
    
    button.addEventListener('mouseover', function() {
        button.style.transform = "scale(1.1)";
        button.style.transition = "all 0.3s ease";
    });
    
    button.addEventListener('mouseout', function() {
        button.style.transform = "scale(1)";
    });
});
