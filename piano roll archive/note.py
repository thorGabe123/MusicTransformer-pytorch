class Note:
    def __init__(self, value, time, velocity, length=-1):
        self.value = value
        self.time = time
        self.length = length
        self.velocity = velocity

    def to_dict(self):
        return {
            "value": self.value,
            "time": self.time,
            "length": self.length,
            "velocity": self.velocity,
        }
    
# class Note:
#     def __init__(self, value, time, velocity, channel=1, length=-1):
#         self.value = value
#         self.time = time
#         self.velocity = velocity
#         self.channel = channel
#         self.length = length

#     def to_dict(self):
#         return {
#             "value": self.value,
#             "time": self.time,
#             "velocity": self.velocity,
#             "channel": self.channel,
#             "length": self.length
#         }