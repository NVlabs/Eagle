# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
INTERVAL_START_TOKEN = '<interval>'
INTERVAL_END_TOKEN = '</interval>'
TEXT_MASK_TOKEN = '<text_mask>'
NULL_TOKEN = '<null>'
OOF_TOKEN = '<oof>'
SEP_TOKEN = '</c>'

special_tokens_list = [
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN, IMG_END_TOKEN,
    BOX_START_TOKEN, BOX_END_TOKEN,
    QUAD_START_TOKEN, QUAD_END_TOKEN,
    REF_START_TOKEN, REF_END_TOKEN,
    INTERVAL_START_TOKEN, INTERVAL_END_TOKEN,
    TEXT_MASK_TOKEN,
    NULL_TOKEN,
    SEP_TOKEN,
] 

# Eagle Embodied adds OOF_TOKEN before SEP_TOKEN. LocateAnything keeps
# special_tokens_list above, so its tokenizer vocabulary is unchanged.
eagle_embodied_special_tokens_list = special_tokens_list[:-1] + [OOF_TOKEN, SEP_TOKEN]

# VFlash vocabulary: the <objects> list wrapper and the physical-atom structure tokens.
OBJECTS_START_TOKEN = '<objects>'
OBJECTS_END_TOKEN = '</objects>'
POINT_START_TOKEN = '<point>'
POINT_END_TOKEN = '</point>'
DRIVE_2D_START_TOKEN = '<drive_2d>'
DRIVE_START_TOKEN = '<drive>'
DRIVE_END_TOKEN = '</drive>'
ROBOT_SINGLE_2D_START_TOKEN = '<robot_single_2d>'
ROBOT_LEFT_2D_START_TOKEN = '<robot_left_2d>'
ROBOT_RIGHT_2D_START_TOKEN = '<robot_right_2d>'
ROBOT_SINGLE_START_TOKEN = '<robot_single>'
ROBOT_LEFT_START_TOKEN = '<robot_left>'
ROBOT_RIGHT_START_TOKEN = '<robot_right>'
ROBOT_END_TOKEN = '</robot>'
# Dense per-trajectory point atoms: one atom holds one query point's whole
# future path. The 2D and 3D atoms close with their own end tokens so a
# trajectory's 2D+3D pair stays two separable atoms.
DENSE_2D_START_TOKEN = '<dense_2d>'
DENSE_2D_END_TOKEN = '</dense_2d>'
DENSE_START_TOKEN = '<dense>'
DENSE_END_TOKEN = '</dense>'

physical_atom_tokens_list = [
    POINT_START_TOKEN, POINT_END_TOKEN,
    DRIVE_2D_START_TOKEN, DRIVE_START_TOKEN, DRIVE_END_TOKEN,
    ROBOT_SINGLE_2D_START_TOKEN, ROBOT_LEFT_2D_START_TOKEN,
    ROBOT_RIGHT_2D_START_TOKEN, ROBOT_SINGLE_START_TOKEN,
    ROBOT_LEFT_START_TOKEN, ROBOT_RIGHT_START_TOKEN, ROBOT_END_TOKEN,
    DENSE_2D_START_TOKEN, DENSE_2D_END_TOKEN, DENSE_START_TOKEN, DENSE_END_TOKEN,
]

# Added in this order to the Eagle Embodied tokenizer, these tokens get the ids
# of the paper VFlash checkpoints (<objects> 152685 ... </dense> 152702).
vflash_special_tokens_list = (
    eagle_embodied_special_tokens_list + [OBJECTS_START_TOKEN, OBJECTS_END_TOKEN] + physical_atom_tokens_list
)

number_tokens_list = [f'<{i}>' for i in range(1001)]
