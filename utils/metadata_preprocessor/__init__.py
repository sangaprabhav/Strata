"""
Metadata Preprocessor Package for Dermatology VQA Dataset Creation
"""

from .base_processor import BaseProcessor
from .bcn20k_processor import BCN20KProcessor
from .ddi_processor import DDIProcessor
from .ddi2_processor import DDI2Processor
from .derm12345_processor import Derm12345Processor
from .ham10k_processor import HAM10KProcessor
from .hiba_processor import HIBAProcessor
from .isic2020_processor import ISIC2020Processor
from .mra_midas_processor import MRAMIDASProcessor
from .mskcc_processor import MSKCCProcessor
from .pad_ufes20_processor import PADUfes20Processor
from .patch16_processor import Patch16Processor
from .scin_processor import SCINProcessor

__all__ = [
    'BaseProcessor',
    'BCN20KProcessor',
    'DDIProcessor',
    'DDI2Processor',
    'Derm12345Processor',
    'HAM10KProcessor',
    'HIBAProcessor',
    'ISIC2020Processor',
    'MRAMIDASProcessor',
    'MSKCCProcessor',
    'PADUfes20Processor',
    'Patch16Processor',
    'SCINProcessor'
] 