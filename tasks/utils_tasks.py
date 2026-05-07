# -*- coding: utf-8 -*-

from tasks.atis import TaskAtis
from tasks.trec import TaskTrec
from tasks.awc import TaskAWC

from tasks.snips import TaskSnips
from tasks.top import TaskTop
from tasks.acl_cite import TaskAclCite

from tasks.clinc import TaskClinc
from tasks.facebook import TaskFacebook
from tasks.twacs import TaskTwACS
from tasks.mantis import TaskMantis
from tasks.sci_cite import TaskSciCite

from tasks.banking77 import TaskBanking77
from tasks.slurp import TaskSlurp
from tasks.acid import TaskACID
from tasks.mcid import TaskMCID
from tasks.mix_atis import TaskMixATIS
from tasks.mix_snips import TaskMixSNIPS
from tasks.empathetic_intents import TaskEmpatheticIntents
from tasks.hint3 import TaskHint3
from tasks.dstc8_sgd import TaskDSTC8SGD
from tasks.multiwoz22 import TaskMultiWOZ22

from tasks.multiwoz23 import TaskMultiWOZ23
from tasks.hwu import TaskHWU
from tasks.stanfordlu import TaskStanfordLU
from tasks.mtop import TaskMTOP
from tasks.xsid import TaskXSID
from tasks.minds14 import TaskMinds14
from tasks.conda import TaskConda
from tasks.policyie import TaskPolicyIE
from tasks.moral_stories import TaskMoralStories

from tasks.nlupp import TaskNLUPP
from tasks.plead import TaskPLEAD
from tasks.iterater import TaskIterater
from tasks.arxiv_edits import TaskArxivEdits

from tasks.credit16 import TaskCredit16
from tasks.vira import TaskVIRA
from tasks.dstc11_t2 import TaskDSTC11T2

from tasks.blendx import TaskBlendX
from tasks.urs import TaskURS
from tasks.intent_conan import TaskIntentConan
from tasks.intention_qa import TaskIntentionQA
from tasks.re3_sci2 import TaskRe3Sci2

from tasks.ioinst import TaskIoInst
from tasks.mathdial import TaskMathDial
from tasks.dyndst import TaskDynDST
from tasks.propa_gaze import TaskPropaGaze

from tasks.recap import TaskRecap
from tasks.malint import TaskMalInt
from tasks.i2hate import TaskI2Hate


TASK_CLASS_DICT = {
    "atis": TaskAtis,
    "trec": TaskTrec,
    "awc": TaskAWC,

    "snips": TaskSnips,
    "top": TaskTop,
    "acl_cite": TaskAclCite,

    "clinc": TaskClinc,
    "facebook": TaskFacebook,
    "twacs": TaskTwACS,
    "mantis": TaskMantis,
    "sci_cite": TaskSciCite,

    "banking77": TaskBanking77,
    "slurp": TaskSlurp,
    "acid": TaskACID,
    "mcid": TaskMCID,
    "mix_atis": TaskMixATIS,
    "mix_snips": TaskMixSNIPS,
    "empathetic_intents": TaskEmpatheticIntents,
    "hint3": TaskHint3,
    "dstc8_sgd": TaskDSTC8SGD,
    "multiwoz22": TaskMultiWOZ22,

    "multiwoz23": TaskMultiWOZ23,
    "hwu": TaskHWU,
    "stanfordlu": TaskStanfordLU,
    "mtop": TaskMTOP,
    "xsid": TaskXSID,
    "minds14": TaskMinds14,
    "conda": TaskConda,
    "policyie": TaskPolicyIE,
    "moral_stories": TaskMoralStories,

    "nlupp": TaskNLUPP,
    "plead": TaskPLEAD,
    "iterater": TaskIterater,
    "arxiv_edits": TaskArxivEdits,

    "credit16": TaskCredit16,
    "vira": TaskVIRA,
    "dstc11_t2": TaskDSTC11T2,

    "blendx": TaskBlendX,
    "urs": TaskURS,
    "intent_conan": TaskIntentConan,
    "intention_qa": TaskIntentionQA,
    "re3_sci2": TaskRe3Sci2,
    
    "ioinst": TaskIoInst,
    "mathdial": TaskMathDial,
    "dyndst": TaskDynDST,
    "propa_gaze": TaskPropaGaze,

    "recap": TaskRecap,
    "malint": TaskMalInt,
    "i2hate": TaskI2Hate,
}
