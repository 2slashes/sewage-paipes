from typing import Optional

PipeType = tuple[bool, bool, bool, bool]
Assignment = list[PipeType]
PartialAssignment = list[Optional[PipeType]]
