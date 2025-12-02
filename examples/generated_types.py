# Auto-generated type definitions

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import json

def parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return value
    return value

@dataclass
class SleepinessItem:
    sequenceId: int = None
    studyDeploymentId: str = None
    deviceRoleName: str = None
    measurement: Measurement = None
    triggerIds: List[int] = None
    syncPoint: SyncPoint = None
    dataStream: DataStream = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        if not isinstance(obj, dict): return obj
        instance = cls()
        val = obj.get('sequenceId')
        instance.sequenceId = val
        val = obj.get('studyDeploymentId')
        instance.studyDeploymentId = val
        val = obj.get('deviceRoleName')
        instance.deviceRoleName = val
        val = obj.get('measurement')
        if val is not None:
            instance.measurement = Measurement.from_dict(val)
        val = obj.get('triggerIds')
        instance.triggerIds = val
        val = obj.get('syncPoint')
        if val is not None:
            instance.syncPoint = SyncPoint.from_dict(val)
        val = obj.get('dataStream')
        if val is not None:
            instance.dataStream = DataStream.from_dict(val)
        return instance

@dataclass
class Measurement:
    sensorStartTime: int = None
    data: Data = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        if not isinstance(obj, dict): return obj
        instance = cls()
        val = obj.get('sensorStartTime')
        instance.sensorStartTime = val
        val = obj.get('data')
        if val is not None:
            instance.data = Data.from_dict(val)
        return instance

@dataclass
class Data:
    __type: str = None
    period: int = None
    deviceType: str = None
    deviceRoleName: str = None
    batteryLevel: int = None
    batteryStatus: str = None
    screenEvent: str = None
    type_: str = None
    confidence: int = None
    triggerId: int = None
    taskName: str = None
    destinationDeviceRoleName: str = None
    control: str = None
    steps: int = None
    time: str = None
    speed: float = None
    isMock: bool = None
    heading: float = None
    accuracy: float = None
    altitude: float = None
    latitude: float = None
    longitude: float = None
    speedAccuracy: float = None
    headingAccuracy: float = None
    verticalAccuracy: float = None
    elapsedRealtimeNanos: int = None
    elapsedRealtimeUncertaintyNanos: float = None
    date: str = None
    sunset: str = None
    country: str = None
    sunrise: str = None
    tempMax: float = None
    tempMin: float = None
    areaName: str = None
    humidity: float = None
    pressure: float = None
    windSpeed: float = None
    cloudiness: float = None
    windDegree: float = None
    temperature: float = None
    weatherMain: str = None
    weatherDescription: str = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        if not isinstance(obj, dict): return obj
        instance = cls()
        val = obj.get('__type')
        instance.__type = val
        val = obj.get('period')
        instance.period = val
        val = obj.get('deviceType')
        instance.deviceType = val
        val = obj.get('deviceRoleName')
        instance.deviceRoleName = val
        val = obj.get('batteryLevel')
        instance.batteryLevel = val
        val = obj.get('batteryStatus')
        instance.batteryStatus = val
        val = obj.get('screenEvent')
        instance.screenEvent = val
        val = obj.get('type')
        instance.type_ = val
        val = obj.get('confidence')
        instance.confidence = val
        val = obj.get('triggerId')
        instance.triggerId = val
        val = obj.get('taskName')
        instance.taskName = val
        val = obj.get('destinationDeviceRoleName')
        instance.destinationDeviceRoleName = val
        val = obj.get('control')
        instance.control = val
        val = obj.get('steps')
        instance.steps = val
        val = obj.get('time')
        instance.time = val
        val = obj.get('speed')
        instance.speed = val
        val = obj.get('isMock')
        instance.isMock = val
        val = obj.get('heading')
        instance.heading = val
        val = obj.get('accuracy')
        instance.accuracy = val
        val = obj.get('altitude')
        instance.altitude = val
        val = obj.get('latitude')
        instance.latitude = val
        val = obj.get('longitude')
        instance.longitude = val
        val = obj.get('speedAccuracy')
        instance.speedAccuracy = val
        val = obj.get('headingAccuracy')
        instance.headingAccuracy = val
        val = obj.get('verticalAccuracy')
        instance.verticalAccuracy = val
        val = obj.get('elapsedRealtimeNanos')
        instance.elapsedRealtimeNanos = val
        val = obj.get('elapsedRealtimeUncertaintyNanos')
        instance.elapsedRealtimeUncertaintyNanos = val
        val = obj.get('date')
        instance.date = val
        val = obj.get('sunset')
        instance.sunset = val
        val = obj.get('country')
        instance.country = val
        val = obj.get('sunrise')
        instance.sunrise = val
        val = obj.get('tempMax')
        instance.tempMax = val
        val = obj.get('tempMin')
        instance.tempMin = val
        val = obj.get('areaName')
        instance.areaName = val
        val = obj.get('humidity')
        instance.humidity = val
        val = obj.get('pressure')
        instance.pressure = val
        val = obj.get('windSpeed')
        instance.windSpeed = val
        val = obj.get('cloudiness')
        instance.cloudiness = val
        val = obj.get('windDegree')
        instance.windDegree = val
        val = obj.get('temperature')
        instance.temperature = val
        val = obj.get('weatherMain')
        instance.weatherMain = val
        val = obj.get('weatherDescription')
        instance.weatherDescription = val
        return instance

@dataclass
class SyncPoint:
    synchronizedOn: str = None
    sensorTimestampAtSyncPoint: int = None
    relativeClockSpeed: float = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        if not isinstance(obj, dict): return obj
        instance = cls()
        val = obj.get('synchronizedOn')
        instance.synchronizedOn = val
        val = obj.get('sensorTimestampAtSyncPoint')
        instance.sensorTimestampAtSyncPoint = val
        val = obj.get('relativeClockSpeed')
        instance.relativeClockSpeed = val
        return instance

@dataclass
class DataStream:
    studyDeploymentId: str = None
    deviceRoleName: str = None
    dataType: DataType = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        if not isinstance(obj, dict): return obj
        instance = cls()
        val = obj.get('studyDeploymentId')
        instance.studyDeploymentId = val
        val = obj.get('deviceRoleName')
        instance.deviceRoleName = val
        val = obj.get('dataType')
        if val is not None:
            instance.dataType = DataType.from_dict(val)
        return instance

@dataclass
class DataType:
    namespace: str = None
    name: str = None

    @classmethod
    def from_dict(cls, obj: Any) -> Any:
        if not isinstance(obj, dict): return obj
        instance = cls()
        val = obj.get('namespace')
        instance.namespace = val
        val = obj.get('name')
        instance.name = val
        return instance
