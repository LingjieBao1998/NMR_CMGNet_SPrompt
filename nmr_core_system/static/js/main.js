// Main JavaScript file for NMR Core Detection System

// Page loading animation (disabled)
// $(window).on('load', function() {
//     // Hide loading overlay when page is fully loaded
//     setTimeout(function() {
//         $('#loading-overlay').addClass('hidden');
//         
//         // Remove from DOM after transition
//         setTimeout(function() {
//             $('#loading-overlay').remove();
//         }, 500);
//     }, 1500); // Show loading animation for at least 1.5 seconds
// });

$(document).ready(function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        $('.alert').fadeOut('slow');
    }, 5000);
    
    // Confirm before destructive actions
    $('.btn-delete').on('click', function(e) {
        if (!confirm('Are you sure you want to delete this item?')) {
            e.preventDefault();
            return false;
        }
    });
    
    // Smooth scrolling for anchor links
    $('a[href*="#"]:not([href="#"])').on('click', function(e) {
        if (location.pathname.replace(/^\//, '') === this.pathname.replace(/^\//, '') && 
            location.hostname === this.hostname) {
            
            var target = $(this.hash);
            target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
            
            if (target.length) {
                e.preventDefault();
                $('html, body').animate({
                    scrollTop: target.offset().top - 70
                }, 1000);
                return false;
            }
        }
    });
    
    // Memgraph Style Carousel Functionality
    var carouselTabs = $('.carousel-tab');
    var carouselSlides = $('.carousel-slide');
    var carouselSlidesContainer = $('.carousel-slides');
    var currentSlide = 0;
    var totalSlides = carouselSlides.length;
    var autoPlayInterval;

    // Function to show specific slide
    function showSlide(index) {
        if (totalSlides === 0) return; // No slides to show

        if (index >= totalSlides) index = 0;
        if (index < 0) index = totalSlides - 1;

        currentSlide = index;

        // Update tabs if they exist
        if (carouselTabs.length) {
            carouselTabs.removeClass('active').attr('aria-pressed', 'false');
            carouselTabs.eq(index).addClass('active').attr('aria-pressed', 'true');
        }

        // Update slides
        carouselSlides.removeClass('active');
        carouselSlides.eq(index).addClass('active');

        // Move slides container
        if (carouselSlidesContainer.length) {
            var translateValue = -(index * 100);
            carouselSlidesContainer.css('transform', 'translate3d(' + translateValue + '%, 0px, 0px)');
        }
    }

    // Tab click handlers (if tabs exist)
    if (carouselTabs.length) {
        carouselTabs.on('click', function() {
            var index = $(this).data('index');
            showSlide(index);

            // Reset auto-play on manual interaction
            clearInterval(autoPlayInterval);
            startAutoPlay();
        });
    }

    // Auto-play functionality
    function startAutoPlay() {
        autoPlayInterval = setInterval(function() {
            showSlide(currentSlide + 1);
        }, 4000); // Change slide every 4 seconds
    }

    // Start auto-play if carousel exists
    if ($('.carousel-slide').length) {
        startAutoPlay();

        // Pause auto-play on hover for carousel container
        $('.carousel-container, .carousel-slides-wrapper').on('mouseenter', function() {
            clearInterval(autoPlayInterval);
        }).on('mouseleave', function() {
            startAutoPlay();
        });
    }
    
    // Loading state for forms
    $('form').on('submit', function() {
        var form = $(this);
        var submitBtn = form.find('button[type="submit"]');
        
        if (form.data('loading-state')) {
            submitBtn.prop('disabled', true);
            submitBtn.html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...');
        }
    });
    
    // Auto-refresh queue stats
    if ($('.task-row').length > 0) {
        setInterval(refreshQueueStats, 10000); // Every 10 seconds
    }
    
    // File input validation
    $('input[type="file"]').on('change', function() {
        var input = $(this);
        var maxSize = 10 * 1024 * 1024; // 10MB
        var allowedTypes = ['application/octet-stream', 'chemical/x-mdl-molfile', 'chemical/x-pdb', 'text/plain'];
        
        if (this.files && this.files[0]) {
            var file = this.files[0];
            
            // Check file size
            if (file.size > maxSize) {
                alert('File size exceeds 10MB limit. Please choose a smaller file.');
                input.val('');
                return false;
            }
            
            // Display file info
            var fileInfo = file.name + ' (' + formatFileSize(file.size) + ')';
            var fileInfoElement = input.siblings('.file-info');
            
            if (fileInfoElement.length) {
                fileInfoElement.text(fileInfo);
            } else {
                input.after('<small class="file-info text-muted">' + fileInfo + '</small>');
            }
        }
    });
    
    // Search functionality
    var searchInput = $('#searchInput');
    var searchTimeout;
    
    searchInput.on('keyup', function() {
        var query = $(this).val();
        var url = $(this).data('search-url');
        
        clearTimeout(searchTimeout);
        
        if (query.length >= 2) {
            searchTimeout = setTimeout(function() {
                performSearch(url, query);
            }, 500);
        }
    });
    
    // Progress bar animation
    $('.progress-bar').each(function() {
        var progressBar = $(this);
        var width = progressBar.attr('aria-valuenow') + '%';
        
        setTimeout(function() {
            progressBar.css('width', width);
        }, 100);
    });
    
    // Status badge colors
    $('.status-badge').each(function() {
        var badge = $(this);
        var status = badge.data('status').toLowerCase();
        
        badge.removeClass('bg-secondary');
        
        if (status === 'completed') {
            badge.addClass('bg-success');
        } else if (status === 'processing') {
            badge.addClass('bg-warning');
        } else if (status === 'failed') {
            badge.addClass('bg-danger');
        } else {
            badge.addClass('bg-secondary');
        }
    });
});

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function refreshQueueStats() {
    $.get('/api/queue-stats/', function(data) {
        // Update statistics displays
        $('.stat-pending').text(data.pending);
        $('.stat-processing').text(data.processing);
        $('.stat-completed').text(data.completed);
        $('.stat-failed').text(data.failed);
        $('.stat-total').text(data.total);
    }).fail(function() {
        console.log('Failed to refresh queue stats');
    });
}

function performSearch(url, query) {
    $.get(url, {q: query}, function(data) {
        // Handle search results
        console.log('Search results:', data);
    }).fail(function() {
        console.log('Search failed');
    });
}

// Task status monitoring
function monitorTaskStatus(taskId, callback) {
    var interval = setInterval(function() {
        $.get('/api/task-status/' + taskId + '/', function(data) {
            if (typeof callback === 'function') {
                callback(data);
            }
            
            // Stop monitoring if task is completed or failed
            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(interval);
            }
        }).fail(function() {
            clearInterval(interval);
        });
    }, 3000); // Check every 3 seconds
    
    return interval;
}

// Auto-resize textarea
$('textarea').on('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // Show success message
        var toast = $('<div class="toast-notification">Copied to clipboard!</div>');
        $('body').append(toast);
        
        setTimeout(function() {
            toast.fadeOut(function() {
                $(this).remove();
            });
        }, 2000);
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
    });
}

// Format date/time
function formatDateTime(dateString) {
    var date = new Date(dateString);
    return date.toLocaleString();
}

// Time ago formatter
function timeAgo(dateString) {
    var date = new Date(dateString);
    var seconds = Math.floor((new Date() - date) / 1000);
    
    var interval = Math.floor(seconds / 31536000);
    if (interval >= 1) return interval + " year" + (interval > 1 ? "s" : "") + " ago";
    
    interval = Math.floor(seconds / 2592000);
    if (interval >= 1) return interval + " month" + (interval > 1 ? "s" : "") + " ago";
    
    interval = Math.floor(seconds / 86400);
    if (interval >= 1) return interval + " day" + (interval > 1 ? "s" : "") + " ago";
    
    interval = Math.floor(seconds / 3600);
    if (interval >= 1) return interval + " hour" + (interval > 1 ? "s" : "") + " ago";
    
    interval = Math.floor(seconds / 60);
    if (interval >= 1) return interval + " minute" + (interval > 1 ? "s" : "") + " ago";
    
    return "just now";
}

// Export functions for global use
window.monitorTaskStatus = monitorTaskStatus;
window.copyToClipboard = copyToClipboard;
window.formatDateTime = formatDateTime;
window.timeAgo = timeAgo;