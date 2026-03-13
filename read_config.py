import configparser


def read_config(configfile):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(configfile)

    # Access values from the configuration file
    last_game = config.getint('General', 'last game')

    learning_rate = config.getfloat('Agent', 'learning rate')
    gamma = config.getfloat('Agent', 'gamma')
    state = config.get('Agent', 'state')

    number_of_colors = config.getint('Game', 'number of colors')
    grid_dimension_1 = config.getint('Game', 'grid dimension 1')
    grid_dimension_2 = config.getint('Game', 'grid dimension 2')

    hidden_size = config.getint('Model', 'hidden size')


    # Return a dictionary with the retrieved values
    config_values = {
        'last game': last_game,
        'learning rate': learning_rate,
        'gamma': gamma,
        'state': state,

        'number of colors': number_of_colors,
        'grid dimension 1': grid_dimension_1,
        'grid dimension 2': grid_dimension_2,

        'hidden size': hidden_size
    }

    return config_values
    

if __name__ == "__main__":
    # Call the function to read the configuration file
    config_data = read_config()