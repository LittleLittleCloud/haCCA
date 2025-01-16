from .alignment import direct_alignment
from .utils import _generate_mock_data, _plot_alignment_result

def test_plot_alignment_result():
    # Create mock data
    labels_a = ["1", "2"]
    labels_b = ["1", "2"]
    a = _generate_mock_data(5, 3, labels_a)
    b_prime = _generate_mock_data(8, 2, labels_b)

    # Call direct_alignment
    aligned_b = direct_alignment(a, b_prime, work_dir=None, enable_center_and_scale=False)

    # Call plot_alignment_result
    _plot_alignment_result(a, b_prime, aligned_b, pi=None, title="Transport Plan", work_dir=None, show=True)