from .body_spec import JOINTS as BODY, LINKS as BODY_LINKS
JOINTS=BODY+tuple(f'{s}_hand_{p}_joint' for s in ('left','right') for p in ('thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1'))
LINKS=BODY_LINKS+tuple(f'{s}_hand_{p}_finger_tip' for s in ('left','right') for p in ('thumb','middle','index'))
ACTOR_DIM=244
CRITIC_DIM=437
GROUPS=(range(0,6),range(6,12),range(12,15),range(15,22),range(22,29),range(29,36),range(36,43))
