JS = """
function start() {
    document.querySelectorAll('.tree').forEach(function(tree) {
        tree.addEventListener('click', function(event) {
            if (event.target.classList.contains('caret')) {
                const path = event.target.parentElement.getAttribute('data-path');
                const treeId = tree.getAttribute('id');
                const pathInput = document.querySelector(`#path_input_${treeId}`);
                if (pathInput) {
                    const textarea = pathInput.querySelector('textarea');
                    textarea.value = path;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    textarea.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        });
    });
}
"""