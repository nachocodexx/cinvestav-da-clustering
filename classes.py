class Topic(object):
    def __init__(self,node_id,role,topic_id,volume,density,files):
        self.node_id=node_id
        self.role=role
        self.topic_id =topic_id
        self.volume=volume
        self.density=density
        self.files=files
class File(object):
    def __init__(self,node_id,file_id,size):
        self.node_id=node_id
        self.file_id=file_id
        self.size=size
class Producer(object):
    def __init__(self,node_id,volume,role,density,production,user_id,consumer_others,consume):
        self.node_id =node_id
        self.volume=volume
        self.role=role
        self.density=density
        self.production=production
        self.user_id=user_id
        self.consume_others=consumer_others
        self.consume=consume
