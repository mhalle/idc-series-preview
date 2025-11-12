"""
DICOM Slice Sorting and Parameterization

Provides centralized logic for sorting DICOM slices according to radiological
conventions and parameterizing them with t ∈ [0, 1] using uniform index spacing.

The sort_slices() function automatically detects which axis (X, Y, or Z) varies
most across slices and sorts by that axis in standard radiological viewing order:
- Axial (Z varies): Superior to Inferior (head to feet, decreasing Z)
- Sagittal (X varies): Right to Left (patient's right to left, decreasing X)
- Coronal (Y varies): Anterior to Posterior (front to back, decreasing Y)

The SliceParameterization class enables parametric navigation through slices
using a continuous parameter t ∈ [0, 1], useful for interactive viewers and
smooth slice selection.

Author: M (original design), Claude (integration)
Date: 2025-11-10
"""

from typing import List, Dict, Any

SliceDict = Dict[str, Any]


def sort_slices(slices: List[SliceDict]) -> List[SliceDict]:
    """
    Sort slices for radiological viewing.

    Detects which axis (X=0, Y=1, Z=2) varies most across slices and sorts by that.
    Assumes axis-aligned scans (axial, sagittal, or coronal).

    Standard radiological viewing order (as if looking at patient from feet):
    - Axial (Z varies): Superior to Inferior (head to feet, decreasing Z)
    - Sagittal (X varies): Right to Left (patient's right to left, decreasing X)
    - Coronal (Y varies): Anterior to Posterior (front to back, decreasing Y)

    Returns sorted slices with added fields:
    - 'sort_key': which field was used ('ImagePositionPatient' or 'InstanceNumber')
    - 'sort_value': the value used for sorting
    - 'axis': for spatial sorting, which axis (0=X, 1=Y, 2=Z), None for InstanceNumber
    - 'axis_label': human-readable label ('X', 'Y', 'Z', or 'I' for InstanceNumber)

    Args:
        slices: List of slice dictionaries. Each must have either:
            - 'ImagePositionPatient': [x, y, z] coordinates
            - 'InstanceNumber': integer instance number (fallback)

    Returns:
        Sorted list of slice dictionaries with sorting metadata added.

    Raises:
        ValueError: If slices list is empty.
    """
    if not slices:
        raise ValueError("Cannot sort empty slices list")

    # First check if we have spatial info
    has_spatial = any("ImagePositionPatient" in s for s in slices)

    if has_spatial:
        # Get all positions
        positions = []
        for s in slices:
            if "ImagePositionPatient" in s:
                positions.append(s["ImagePositionPatient"])

        # Find which axis varies most (has largest range)
        # Calculate range for each axis (X=0, Y=1, Z=2)
        ranges = [0, 0, 0]
        for axis_idx in range(3):
            values = [pos[axis_idx] for pos in positions]
            ranges[axis_idx] = max(values) - min(values)

        # Find axis with maximum range
        axis = ranges.index(max(ranges))

        # Determine sort direction for radiological viewing convention
        # Standard radiological viewing (as if looking at patient from their feet):
        # - Axial (Z=2): Superior to Inferior (head to feet) - DECREASING Z
        # - Sagittal (X=0): Right to Left (patient's right to left) - DECREASING X
        # - Coronal (Y=1): Anterior to Posterior (front to back) - DECREASING Y
        # All orientations use decreasing order for standard radiological viewing
        negate = True  # Always negate for standard viewing order

        # Apply to all slices
        for s in slices:
            if "ImagePositionPatient" in s:
                value = s["ImagePositionPatient"][axis]
                s["sort_value"] = -value if negate else value
                s["sort_key"] = "ImagePositionPatient"
                s["axis"] = axis
                s["axis_label"] = ["X", "Y", "Z"][axis]
            else:
                # No position for this slice
                s["sort_value"] = s.get("InstanceNumber", 0)
                s["sort_key"] = "InstanceNumber"
                s["axis"] = None
                s["axis_label"] = "I"  # Instance number
    else:
        # Fallback to instance number for all slices
        for s in slices:
            s["sort_value"] = s.get("InstanceNumber", 0)
            s["sort_key"] = "InstanceNumber"
            s["axis"] = None
            s["axis_label"] = "I"  # Instance number

    return sorted(slices, key=lambda s: s["sort_value"], reverse=True)


class SliceParameterization:
    """
    Parameterize slices with t ∈ [0, 1] using uniform index spacing.

    For n slices: t = i/(n-1) where i is the slice index.

    This enables:
    - Interactive viewers with continuous t parameter
    - Smooth slice navigation
    - Position-independent slice selection

    Usage:
        # Sort slices
        slices_sorted = sort_slices(raw_slices)

        # Parameterize
        param = SliceParameterization(slices_sorted)

        # Get t for slice i
        t = param.get_t(5)

        # Get slice info for t
        info = param.get_slice(0.5)
        print(f"Index: {info['index']}")
        print(f"Sort key: {info['sort_key']}")
        if info['sort_key'] == 'ImagePositionPatient':
            print(f"Position: {info['position']}")
        else:
            print(f"Instance: {info['instance_number']}")
    """

    def __init__(self, slices: List[SliceDict]):
        """
        Initialize with sorted slices.

        Parameters
        ----------
        slices : List[SliceDict]
            Sorted slices (use sort_slices first)
        """
        self.slices = slices
        self.n = len(slices)

        # Uniform index spacing: t = i/(n-1)
        if self.n == 1:
            self.t_values = [0.0]
        else:
            self.t_values = [i / (self.n - 1) for i in range(self.n)]

    def get_t(self, index: int) -> float:
        """
        Get t value for slice at index.

        Parameters
        ----------
        index : int
            Slice index (0-based)

        Returns
        -------
        float
            t value in [0, 1]
        """
        if index < 0 or index >= self.n:
            raise IndexError(f"Index {index} out of range [0, {self.n-1}]")
        return float(self.t_values[index])

    def get_slice(self, t: float) -> Dict[str, Any]:
        """
        Get slice information for given t.

        Parameters
        ----------
        t : float
            Parameter in [0, 1]

        Returns
        -------
        dict
            Dictionary with:
            - 'index': slice index in sorted array
            - 't': actual t value at this slice
            - 'sort_key': 'ImagePositionPatient' or 'InstanceNumber'
            - 'sort_value': the value used for sorting
            - 'axis': axis index if spatial (0=X, 1=Y, 2=Z), None otherwise
            - 'axis_label': human-readable label ('X', 'Y', 'Z', or 'I')
            - 'slice': the full slice dictionary

            Plus one of:
            - 'position': Position coordinate (if sort_key is ImagePositionPatient)
            - 'instance_number': instance number (if sort_key is InstanceNumber)
        """
        if not 0 <= t <= 1:
            raise ValueError(f"t must be in [0, 1], got {t}")

        # Find closest t
        distances = [abs(t_val - t) for t_val in self.t_values]
        idx = distances.index(min(distances))

        slice_dict = self.slices[idx]

        # Build return info
        info = {
            "index": idx,
            "t": float(self.t_values[idx]),
            "sort_key": slice_dict["sort_key"],
            "sort_value": slice_dict["sort_value"],
            "axis": slice_dict.get("axis"),
            "axis_label": slice_dict.get("axis_label", "I"),
            "slice": slice_dict,
        }

        # Add the actual underlying value (not the negated sort_value)
        if info["sort_key"] == "ImagePositionPatient":
            info["position"] = slice_dict["ImagePositionPatient"][info["axis"]]
        elif info["sort_key"] == "InstanceNumber":
            info["instance_number"] = slice_dict.get("InstanceNumber")

        return info

    def get_slices_in_range(
        self, t_start: float, t_end: float
    ) -> List[Dict[str, Any]]:
        """
        Get all slices with t in [t_start, t_end].

        Parameters
        ----------
        t_start : float
            Start of range [0, 1]
        t_end : float
            End of range [0, 1]

        Returns
        -------
        List[dict]
            List of slice info dicts (same format as get_slice)
        """
        if not (0 <= t_start <= 1 and 0 <= t_end <= 1):
            raise ValueError("t_start and t_end must be in [0, 1]")

        indices = [
            i for i, t_val in enumerate(self.t_values) if t_start <= t_val <= t_end
        ]

        return [self.get_slice(self.t_values[idx]) for idx in indices]
