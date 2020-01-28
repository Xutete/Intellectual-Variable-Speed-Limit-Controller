import torch

# Saving model
def save_model(net, path_net, frame):
	torch.save({
		'frame': frame,
		'state_dict': net.state_dict(),
		},
		path_net)

# Load pretrained model
def load_model(net, path_net):
	state_dict = torch.load(path_net)
	net.load_state_dict(state_dict['state_dict'])
	frame = state_dict['frame']
	print("Having previously run %d frames." % frame)
	net.train()
	return net, frame + 1