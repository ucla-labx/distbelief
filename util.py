"""
util functions 
"""

def serialize_tensor(tensor):
    np_tensor = tensor.numpy()
    bytes = np_tensor.tobytes()
    encoded = base64.encodestring(bytes)
    info = (encoded.decode('ascii'), np_tensor.shape, str(np_tensor.dtype))
        
    return json.dumps(info)

def deserialize_tensor(serialized_tensor):
    data, shape, type = json.loads(serialized_tensor)
    tensor = np.frombuffer(base64.decodestring(data.encode('ascii')), type)
    return torch.Tensor(tensor).view(shape)


def test():
    a = torch.rand((2, 4))
    print(a)

    serialized = serialize_tensor(a)
    print(serialized)

    deserialized = deserialize_tensor(serialized)
    print(deserialized)

    assert deserialized.equal(a)

