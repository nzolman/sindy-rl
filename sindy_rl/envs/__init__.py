from .inverted_pendulum import InvPendSurrogate
from .swimmer import SwimmerSurrogate
import warnings

try:
    from .hydroenv import SurrogateCylinder, SurrogatePinball
except ImportError:
    warnings.warn("Hydrogym not found! Can't use hydrogym environments")