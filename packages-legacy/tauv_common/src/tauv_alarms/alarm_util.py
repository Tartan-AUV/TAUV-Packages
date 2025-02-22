from abc import abstractclassmethod
from enum import Enum
import typing

class FailureLevel(Enum):
    NO_FAILURE = 0
    PREDIVE_FAILURE = 1
    MISSION_FAILURE = 2
    CRITICAL_FAILURE = 3


class AlarmType:
    def __init__(self,
            name: str,
            id: int,
            failure_level: FailureLevel,
            default_set: bool,
            description: str,
            author: str):
        self.name = name
        self.id = id
        self.description = description
        self.author = author
        self.failure_level = failure_level
        self.default_set = default_set

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        return False

    def __hash__(self):
        return self.id

    def __str__(self):
        return f"Alarm {self.id}: {self.name}"


# Metaclass that allows for the enum-like behavior of the Alarm class.
class AlarmMeta(type):
    def __call__(cls, id: int) -> AlarmType:
        return cls.__idhash__[id]

    def __new__(mcls, name, bases, namespace, **kw):
        cls =  super().__new__(mcls, name, bases, namespace, **kw)
        cls.__idhash__ = {a.id : a for a in cls}
        return cls

    def __iter__(cls) -> typing.Iterator[AlarmType]:
        yield from [cls.__dict__[k] for k in cls.__dict__.keys() if k[:2] != '__']

