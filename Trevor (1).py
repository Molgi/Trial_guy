import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import math
# import networkx as nx
from collections import defaultdict

# Initilaise variables
min_channel_bw = 5
max_channel_bw = 10
number_channels = 10

min_Tx_power_prim = 10
max_Tx_power_prim = 15
min_migration_cost = 0
max_migration_cost = 7

min_channel_noise=0.5
max_channel_noise=1.5
min_user_bw=1.5 # below this value, the user QoS is said to be violated
def generate_channel_noise_vector(number_channels):
    channel_noise_vector=[]
    for channel_noise in range(number_channels):
        channel_noiz = np.random.uniform(min_channel_noise, max_channel_noise)
        channel_noise_vector.append(channel_noiz)
    return channel_noise_vector

def initialise_secondary_user_placement(numb_sec_users,number_channels):
    # sets the intial_placement of secondary users in the channel. This is randomly done
    sec_user_placement_dict=defaultdict(list)
    channel_IDs=[x for x in range(number_channels)]
    user_IDS=[x  for x in range(numb_sec_users)]
    for user_ID in user_IDS:
        # choose channel for placement of this user randomly.
        channel=np.random.choice(channel_IDs)
        sec_user_placement_dict[channel].append(user_ID)# store user in selected channel
    return sec_user_placement_dict

def generate_channel_bw_vector(number_channels):
    # Generating channel_bandwidths for any number of channels:
    channel_bw_vector = []
    for channel_bw in range(number_channels):
        channel_bw = np.random.uniform(min_channel_bw, max_channel_bw + 1)
        channel_bw_vector.append(channel_bw)
    return channel_bw_vector

def generate_user_prim_bw_demand(number_channels, channel_bw_vector):
    # Generating the bandwidths for the different primary users in each of the alloted channels
    primary_bw_demand_vector = []  # list stores the primary user bandwidths
    for channel_ID in range(number_channels):
        max_bw = channel_bw_vector[channel_ID]
        user_prim_demand = np.random.uniform(0, max_bw + 1)
        primary_bw_demand_vector.append(user_prim_demand)
    return primary_bw_demand_vector

def generate_residual_channel_bw_vector(number_channels, channel_bw_vector, primary_bw_demand_vector):
    # Generating the residual channel bandwidths to be shared between the secondary users for each channel
    residual_channel_bw_vector = []  # list stores the residual channel bandwidth
    for channel_ID in range(number_channels):
        max_bw = channel_bw_vector[channel_ID]
        taken = primary_bw_demand_vector[channel_ID]
        available_bw = max_bw - taken
        residual_channel_bw_vector.append(available_bw)
    return residual_channel_bw_vector

def generate_migration_cost_matrix(min_migration_cost, max_migration_cost, number_channels):
    # Generating the value corresponding to the cost of migrating from one channel to another
    cost_mig_matrix = np.zeros((number_channels, number_channels), float)  # initialising the cost matrix with zeros
    for row in range(number_channels):
        for col in range(number_channels):
            if row == col:
                mig_cost = 0
                cost_mig_matrix[row][col] = mig_cost  # same channel movement is set to zero
            else:
                mig_cost = np.random.uniform(min_migration_cost,
                                             max_migration_cost)  # filling up the migration cost matrix
                cost_mig_matrix[row][col] = mig_cost
    return cost_mig_matrix

def compute_system_data_rate(sec_user_placement_dict, residual_channel_bw_vector, primary_bw_demand_vector,
                             max_Tx_power_prim, channel_noise_vector,min_user_data_rate):
    # initialising the total data_rates
    total_user_sec_data_rate = 0
    total_user_prim_data_rate = 0
    total_num_primary = 0
    total_numb_secondary = 0
    total_QoS_violations=0
    for channel_ID in range(len(residual_channel_bw_vector)):  # iterates through every channel
        channel_noise = channel_noise_vector[channel_ID]
        # assuming uniform datarates for all the primary and secondary users of a particular channel
        if channel_ID in sec_user_placement_dict and len(sec_user_placement_dict[channel_ID])>0:
            numb_sec_users_in_channel_ID=len(sec_user_placement_dict[channel_ID])

            total_numb_secondary += numb_sec_users_in_channel_ID
            bw_per_sec_user = residual_channel_bw_vector[channel_ID] / numb_sec_users_in_channel_ID
            B = bw_per_sec_user
            if numb_sec_users_in_channel_ID >= 1:  # if there is more than a single sec user in a channel
                den_val=(channel_noise + ((numb_sec_users_in_channel_ID - 1) * max_Tx_power_prim))

                sec_user_data_rate = B * math.log10(1 + max_Tx_power_prim / (channel_noise + ((
                                                                                                 numb_sec_users_in_channel_ID - 1) * max_Tx_power_prim)))  # calculating the noise due to the other sec users in the channel without including the said sec user
                total_user_sec_data_rate += sec_user_data_rate * numb_sec_users_in_channel_ID  # i don't think we need to loop through for every user
                if bw_per_sec_user < min_user_bw:
                    total_QoS_violations += numb_sec_users_in_channel_ID

                # we need a break condition for the "if" loop

            # situation where there's only a single sec user in a channel
            else:
                sec_user_data_rate = B * log10(1 + max_Tx_power_prim / (channel_noise))
                total_user_sec_data_rate += sec_user_data_rate * numb_sec_users_in_channel_ID  # I think the += is unnecessary
                if bw_per_sec_user < min_user_bw:
                    total_QoS_violations += 1

            # Computing the primary user datarate for this channel:

            B_primary = primary_bw_demand_vector[channel_ID]
            if B_primary > 0:
                total_num_primary += 1
                prim_user_data_rate = B_primary * math.log10(
                    1 + max_Tx_power_prim / (
                            channel_noise + ((numb_sec_users_in_channel_ID) * max_Tx_power_prim)))
                total_user_prim_data_rate += prim_user_data_rate



    # average values of data rate
    average_data_rate_sec = total_user_sec_data_rate / total_numb_secondary
    average_data_rate_prim = total_user_prim_data_rate / total_num_primary

    return average_data_rate_sec, average_data_rate_prim,total_QoS_violations





def perform_handoff_greedy_primary_users(sec_user_placement_dict, numb_sec_users,primary_bw_demand_vector,cost_mig_matrix):
    # This function explores the greedy approach for a handoff decision based on thethe chnale with least number of primary users
    number_channels=len(primary_bw_demand_vector)
    prev_sec_user_placement_dict = copy.deepcopy(sec_user_placement_dict)  # current user association
    new_sec_user_placement_dict = defaultdict(list)  # store new migration for users
    channel_IDs = [x for x in range(number_channels)]
    user_IDs = [x for x in range(numb_sec_users)]
    for user_ID in user_IDs:
        # map this user in the channel with the list index:
        min_prim_demand=min(primary_bw_demand_vector)
        selected_channel=primary_bw_demand_vector.index(min_prim_demand)
        new_sec_user_placement_dict[selected_channel].append(user_ID)
    numb_handoffs=compute_numb_handoffs(prev_sec_user_placement_dict, new_sec_user_placement_dict, numb_sec_users)
    migration_cost=compute_migration_cost(prev_sec_user_placement_dict, new_sec_user_placement_dict, numb_sec_users,cost_mig_matrix)

    return new_sec_user_placement_dict,numb_handoffs,migration_cost

def perform_handoff_Greedy_migration_cost(migration_cost_vector, number_channels,sec_user_placement_dict,numb_sec_users):
    prev_sec_user_placement_dict=copy.deepcopy(sec_user_placement_dict)# current user association
    new_sec_user_placement_dict=defaultdict(list)# store new migration for users
    channel_IDs=[x for x in range(number_channels)]
    for channel_ID in channel_IDs:
        user_list=prev_sec_user_placement_dict[channel_ID]
        if len(user_list)>0:
           for user in user_list:
               current_channel=channel_ID # channel where the user is currently mapped
               considered_row_cost = migration_cost_vector[current_channel]
               mini_cost = 100000
               for cost_mig_value in considered_row_cost:
                   if cost_mig_value < mini_cost and cost_mig_value != 0:
                       mini_cost = cost_mig_value
               channel_index = list(considered_row_cost).index(mini_cost)
               selected_channel = channel_index
               new_sec_user_placement_dict[selected_channel].append(user)
    numb_handoffs=compute_numb_handoffs(prev_sec_user_placement_dict, new_sec_user_placement_dict, numb_sec_users)
    migration_cost=compute_migration_cost(prev_sec_user_placement_dict, new_sec_user_placement_dict, numb_sec_users, cost_mig_matrix)

    return new_sec_user_placement_dict,numb_handoffs,migration_cost

def compute_numb_handoffs(prev_sec_user_placement_dict,new_sec_user_placement_dict,numb_sec_users):
    # Compute the number of handoff based on previous and current associations
    numb_handoffs=0
    for user_ID in range(numb_sec_users):
        for channel_ID in range(number_channels):
            prev_mapped_user_list=[]
            cur_mapped_user_list=[]
            if channel_ID in prev_sec_user_placement_dict:
                prev_mapped_user_list=prev_sec_user_placement_dict[channel_ID]# users previous mapped in the channel
            if channel_ID in new_sec_user_placement_dict:
                cur_mapped_user_list=new_sec_user_placement_dict[channel_ID]# users current mapped in this channel
            if (user_ID in prev_mapped_user_list and user_ID not in cur_mapped_user_list) or (user_ID in cur_mapped_user_list and user_ID not in prev_mapped_user_list):
                # In this case, the user changed channel
                numb_handoffs+=1

    return numb_handoffs

def compute_migration_cost(prev_sec_user_placement_dict,new_sec_user_placement_dict,numb_sec_users,cost_mig_matrix):
    # Compute the number of handoff based on previous and current associations
    total_mig_cost=0
    for user_ID in range(numb_sec_users):# find where the user is currently placed and where he was before
        previous_channel=None # channel wehre user was previosly served
        Current_Channel=None # where user is currently served
        for channel_ID in range(number_channels):
            if channel_ID in prev_sec_user_placement_dict:
                prev_mapped_user_list=prev_sec_user_placement_dict[channel_ID]# users previous mapped in the channel
                if user_ID in prev_mapped_user_list:
                    previous_channel=channel_ID
            if channel_ID in new_sec_user_placement_dict:
                cur_mapped_user_list=new_sec_user_placement_dict[channel_ID]# users current mapped in this channel
                if user_ID in cur_mapped_user_list:
                    Current_Channel=channel_ID
        if previous_channel==None or Current_Channel==None:
            pass
        else:
            mig_cost=cost_mig_matrix[previous_channel][Current_Channel]
            total_mig_cost+=mig_cost


    return total_mig_cost


def perform_handoff_random(number_channels,sec_user_placement_dict,numb_sec_users, cost_mig_matrix):
    # This function explores the random approach for performing handoff decision of each secondary use
    # sec_user_placement_dict key is the channel index and value is list of users in the channle
    new_sec_user_placement_dict=defaultdict(list) # store the new assoication of users
    prev_sec_user_placement_dict=copy.deepcopy(sec_user_placement_dict)
    channel_list = [x for x in range(number_channels)]
    for chanel_ID in sec_user_placement_dict:
        user_list=sec_user_placement_dict[chanel_ID] 

        if len(user_list)>0:
            for user in user_list:# select the chanel ID randomly
                selected_channel = int(np.random.choice(channel_list, 1)) # new channel
                new_sec_user_placement_dict[selected_channel].append(user)
    #return new_sec_user_placement_dict
    # Compute the number of handoff
    numb_handoffs=compute_numb_handoffs(prev_sec_user_placement_dict, new_sec_user_placement_dict, numb_sec_users)
    migration_cost=compute_migration_cost(prev_sec_user_placement_dict, new_sec_user_placement_dict, numb_sec_users, cost_mig_matrix)
    return new_sec_user_placement_dict,numb_handoffs,migration_cost
def evaluate_Channel_Quality(current_Channel, cand_channel,residual_channel_bw_vector,sec_user_placement_dict,cost_mig_matrix):
    mig_cost=cost_mig_matrix[current_Channel][cand_channel]
    residual_bw=residual_channel_bw_vector[cand_channel]
    num_sec_user=len(sec_user_placement_dict[cand_channel])
    Chanel_score=(residual_bw)/(mig_cost+num_sec_user)
    return Chanel_score

def perform_handoff_Channel_Q(number_channels,sec_user_placement_dict,numb_sec_users,cost_mig_matrix):
    prev_sec_user_placement_dict=copy.deepcopy(sec_user_placement_dict)# current user association
    new_sec_user_placement_dict=defaultdict(list)# store new migration for users
    channel_IDs=[x for x in range(number_channels)]
    mapped_users=[]
    for channel_ID in channel_IDs:
        if channel_ID not in prev_sec_user_placement_dict:
            continue
        user_list=prev_sec_user_placement_dict[channel_ID]
        if len(user_list)>0:
           for user in user_list:
               if user in mapped_users:
                   continue
               current_channel=channel_ID # channel where the user is currently mapped
               # Evaluate chanel quality of possible cadidate channels:
               sorted_channel_lst = sorted(channel_IDs, key=lambda x:
               evaluate_Channel_Quality(current_channel, x, residual_channel_bw_vector,prev_sec_user_placement_dict, cost_mig_matrix), reverse=True)
               selected_channel=sorted_channel_lst[0]
               new_sec_user_placement_dict[selected_channel].append(user)
               mapped_users.append(user)
               # Update the placement dict:
               if selected_channel!=current_channel:# remove user from channel
                   user_list.remove(user)
                   prev_sec_user_placement_dict[current_channel]=user_list
                   prev_sec_user_placement_dict[selected_channel].append(user)
    numb_handoffs = compute_numb_handoffs(sec_user_placement_dict, new_sec_user_placement_dict, numb_sec_users)
    migration_cost = compute_migration_cost(sec_user_placement_dict, new_sec_user_placement_dict, numb_sec_users,
                                            cost_mig_matrix)
    return new_sec_user_placement_dict,numb_handoffs,migration_cost

def reward(state_space, action_space):
    if action == 0:
        return -1
    else:
        if state[1] < 3:
            return 10
        else:
            return -10


def transition(state_space, action_space):
    if action == 'stay':
        return state
    else:
        return (state[0], state[1] + 1)


def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    Q = {}
    for state in env.states:
        for action in env.actions:
            Q[(state, action)] = 0

    for epsiode in range(epsiodes):
        state = random.choice(env.states)
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.actions)
            else:
                values = np.array([Q[(state, a)] for a in env.actions])
                action = env.actions[np.argmax(values)]

            next_state = env.trnsition(state, action)
            r = env.reward(state, action)
            Q[(state, action)] += alpha * (r + gamma * np.max([Q[(next_state, a)] for a in env.actions]) -
                                           Q[(state, action)])
            state = next_state
            if state[1] == 4:
                done = True
    return Q







#######

# Store average values here for plotting
mean_data_rate_vector_QL = []
mean_data_rate_vector_Random = []
mean_data_rate_vector_migC = []
mean_data_rate_vector_G_prim_users = []

mean_migCost_vector_QL = []
mean_migCost_vector_Random = []
mean_migCost_vector_migC = []
mean_migCost_vector_G_prim_users = []

mean_time_vector_QL = []
mean_time_vector_Random = []
mean_time_vector_migC = []
mean_time_vector_G_prim_users = []

mean_HO_vector_QL = []
mean_HO_vector_Random = []
mean_HO_vector_migC = []
mean_HO_vector_G_prim_users = []

mean_violations_vector_QL = []
mean_violations_vector_Random = []
mean_violations_vector_migC = []
mean_violations_vector_G_prim_users = []

###################################
secondary_user_range_list = [5,10,15,20,25,30,35,40]  # list containg the different number of secondary users to be used in simulations
max_iters = 1000
max_trials = 10 # maximum trials to test for each number of secondary users


for numb_sec_users in secondary_user_range_list:  # test perfroamnce for different numbers of secondary users
    numb_secondary_users = numb_sec_users

    sampled_data_rate_vector_QL = []
    sampled_data_rate_vector_Random = []
    sampled_data_rate_vector_migC = []
    sampled_data_rate_vector_G_prim_users = []

    sampled_migCost_vector_QL = []
    sampled_migCost_vector_Random = []
    sampled_migCost_vector_migC = []
    sampled_migCost_vector_G_prim_users = []

    sampled_time_vector_QL = []
    sampled_time_vector_Random = []
    sampled_time_vector_migC = []
    sampled_time_vector_G_prim_users = []

    sampled_HO_vector_QL = []
    sampled_HO_vector_Random = []
    sampled_HO_vector_migC = []
    sampled_HO_vector_G_prim_users = []

    sampled_violations_vector_QL = []
    sampled_violations_vector_Random = []
    sampled_violations_vector_migC = []
    sampled_violations_vector_G_prim_users = []
    total_sec_users=0
    for trial in range(max_trials):
        total_sec_users+=numb_sec_users
        # define channel properties
        # Initilise the placement of secondary users
        sec_user_placement_dict = initialise_secondary_user_placement(numb_sec_users, number_channels)
        channel_bw_vector = generate_channel_bw_vector(number_channels)
        channel_noise_vector=generate_channel_noise_vector(number_channels)
        cost_mig_matrix = generate_migration_cost_matrix(min_migration_cost, max_migration_cost, number_channels)

        total_handoffs_G_prim_users=0
        total_handoffs_Random = 0
        total_handoffs_migC =0
        total_handoffs_QL=0

        avg_data_rate_G_prim_users = []
        avg_data_rate_Random = []
        avg_data_rate_migC = []
        avg_data_rate_QL = []

        total_migration_cost_G_prim_users = 0
        total_migration_cost_Random = 0
        total_migration_cost_migC = 0
        total_migration_cost_QL = 0

        total_QoS_G_prim_users = 0
        total_QoS_Random = 0
        total_QoS_migC = 0
        total_QoS_QL = 0
        sec_user_placement_dict_Random=copy.deepcopy(sec_user_placement_dict)
        sec_user_placement_dict_migC = copy.deepcopy(sec_user_placement_dict)
        sec_user_placement_dict_G_prim_users = copy.deepcopy(sec_user_placement_dict)
        sec_user_placement_dict_QL=copy.deepcopy(sec_user_placement_dict)
        for itern in range(max_iters):# perform handoffs here
            # generate primary users
            primary_bw_demand_vector = generate_user_prim_bw_demand(number_channels, channel_bw_vector)
            residual_channel_bw_vector = generate_residual_channel_bw_vector(number_channels, channel_bw_vector,
                                                                             primary_bw_demand_vector)
            # decide to perform handoff
            # Call random algorithm##########
            sec_user_placement_dict_Random,numb_handoffs_Random,mig_cost_Random=perform_handoff_random(number_channels,sec_user_placement_dict_Random,numb_sec_users,cost_mig_matrix)
            #compute resulting data rate from this handoff procedure
            average_data_rate_sec_Random, average_data_rate_prim_Random,QoS_violations_Random = compute_system_data_rate(
                sec_user_placement_dict_Random, residual_channel_bw_vector, primary_bw_demand_vector,
                max_Tx_power_prim, channel_noise_vector,min_user_bw)
            # Update and store vaues
            avg_data_rate_Random.append(average_data_rate_sec_Random)
            total_QoS_Random+=QoS_violations_Random
            total_migration_cost_Random+=mig_cost_Random
            total_handoffs_Random+=(numb_handoffs_Random)


            # CALL MIG COST BASED APPROACH

            sec_user_placement_dict_migC,numb_handoffs_migC,mig_cost_migC=perform_handoff_Greedy_migration_cost(cost_mig_matrix, number_channels, sec_user_placement_dict_migC,
                                                  numb_sec_users)

            # compute resulting data rate from this handoff procedure
            average_data_rate_sec_migC, average_data_rate_prim_migC, QoS_violations_migC = compute_system_data_rate(
                sec_user_placement_dict_migC, residual_channel_bw_vector, primary_bw_demand_vector,
                max_Tx_power_prim, channel_noise_vector, min_user_bw)
            # Update and store vaues
            avg_data_rate_migC.append(average_data_rate_sec_migC)
            total_QoS_migC += QoS_violations_migC
            total_migration_cost_migC += mig_cost_migC
            total_handoffs_migC += (numb_handoffs_migC)

            # Compute greedy primary users
            sec_user_placement_dict_G_prim_users,numb_handoffs_G_prim_users,mig_cost_G_prim=perform_handoff_greedy_primary_users(sec_user_placement_dict_G_prim_users, numb_sec_users, primary_bw_demand_vector,cost_mig_matrix)

            # compute resulting data rate from this handoff procedure
            average_data_rate_sec_G_prim, average_data_rate_prim_G_prim, QoS_violations_G_prim = compute_system_data_rate(
                sec_user_placement_dict_G_prim_users, residual_channel_bw_vector, primary_bw_demand_vector,
                max_Tx_power_prim, channel_noise_vector, min_user_bw)
            # Update and store vaues
            avg_data_rate_G_prim_users.append(average_data_rate_sec_G_prim)
            total_QoS_G_prim_users += QoS_violations_G_prim
            total_migration_cost_G_prim_users += mig_cost_G_prim
            total_handoffs_G_prim_users += (numb_handoffs_G_prim_users)

            # CALL THE QBASED ALGORITHM
            sec_user_placement_dict_QL,numb_handoffs_Q,mig_cost_Q=perform_handoff_Channel_Q(number_channels, sec_user_placement_dict_QL, numb_sec_users, cost_mig_matrix)

            # compute resulting data rate from this handoff procedure
            average_data_rate_sec_QL, average_data_rate_prim_QL, QoS_violations_QL = compute_system_data_rate(
                sec_user_placement_dict_QL, residual_channel_bw_vector, primary_bw_demand_vector,
                max_Tx_power_prim, channel_noise_vector, min_user_bw)
            # Update and store vaues
            avg_data_rate_QL.append(average_data_rate_sec_QL)
            total_QoS_QL += QoS_violations_QL
            total_migration_cost_QL += mig_cost_Q
            total_handoffs_QL += (numb_handoffs_Q)





        #store values here

        sampled_data_rate_vector_Random.append(np.mean(avg_data_rate_Random))
        sampled_data_rate_vector_migC.append(np.mean(avg_data_rate_migC))
        sampled_data_rate_vector_G_prim_users.append(np.mean(avg_data_rate_G_prim_users))
        sampled_data_rate_vector_QL.append(np.mean(avg_data_rate_QL))

        sampled_migCost_vector_QL.append(total_migration_cost_QL/total_sec_users)
        sampled_migCost_vector_Random.append(total_migration_cost_Random/total_sec_users)
        sampled_migCost_vector_migC.append(total_migration_cost_migC/total_sec_users)
        sampled_migCost_vector_G_prim_users.append(total_migration_cost_G_prim_users/total_sec_users)



        sampled_HO_vector_Random.append(total_handoffs_Random/(max_iters*total_sec_users))
        sampled_HO_vector_migC.append(total_handoffs_migC/(max_iters*total_sec_users))
        sampled_HO_vector_G_prim_users.append(total_handoffs_G_prim_users/(max_iters*total_sec_users))
        sampled_HO_vector_QL.append(total_handoffs_QL / (max_iters * total_sec_users))


        sampled_violations_vector_Random.append(total_QoS_Random/(max_iters*total_sec_users))
        sampled_violations_vector_migC.append(total_QoS_migC/(max_iters*total_sec_users))
        sampled_violations_vector_G_prim_users.append(total_QoS_G_prim_users/(max_iters*total_sec_users))
        sampled_violations_vector_QL.append(total_QoS_QL / (max_iters * total_sec_users))

    # Store average values here for plotting

    mean_data_rate_vector_Random.append(np.mean(sampled_data_rate_vector_Random))
    mean_data_rate_vector_migC.append(np.mean(sampled_data_rate_vector_migC))
    mean_data_rate_vector_G_prim_users.append(np.mean(sampled_data_rate_vector_G_prim_users))
    mean_data_rate_vector_QL.append(np.mean(sampled_data_rate_vector_QL))


    mean_migCost_vector_Random.append(np.mean(sampled_migCost_vector_Random))
    mean_migCost_vector_migC.append(np.mean(sampled_migCost_vector_migC))
    mean_migCost_vector_G_prim_users.append(np.mean(sampled_migCost_vector_G_prim_users))
    mean_migCost_vector_QL.append(np.mean(sampled_migCost_vector_QL))


    mean_HO_vector_Random.append(np.mean(sampled_HO_vector_Random))
    mean_HO_vector_migC.append(np.mean(sampled_HO_vector_migC))
    mean_HO_vector_G_prim_users.append(np.mean(sampled_HO_vector_G_prim_users))
    mean_HO_vector_QL.append(np.mean(sampled_HO_vector_QL))

    mean_violations_vector_Random.append(np.mean(sampled_violations_vector_Random))
    mean_violations_vector_migC.append(np.mean(sampled_violations_vector_migC))
    mean_violations_vector_G_prim_users.append(np.mean(sampled_violations_vector_G_prim_users))
    mean_violations_vector_QL.append(np.mean(sampled_violations_vector_QL))










# PLOT the results below
x = secondary_user_range_list
plt.figure(1)
plt.title("Average User Datarate")
plt.grid(b=True, which='major', axis='both')
# plt.ylim(0,1)
plt.grid(color='k', linestyle='-', linewidth=0.1)
loc, labels = plt.yticks()
plt.xlim(x[0], x[-1] + 1)
plt.xlabel("Number of Secondary Users")
plt.ylabel("Average user Data rate")
plt.plot(x, mean_data_rate_vector_Random, 'r-p', label='Random')
# plt.plot(x, mean_value_accuracy_HS, '-m^', label='HS')
#plt.plot(x, mean_data_rate_vector_migC, 'k-D', label='GPU')
plt.plot(x, mean_data_rate_vector_G_prim_users, 'g-.s', label='G_prim_users')
plt.plot(x, mean_data_rate_vector_QL, '-b^', label='DRL')
# plt.plot(x, mean_value_accuracy_distance, 'k-.s', label='LB')
# plt.figlegend(('Dist', 'HS', 'Energ', 'Random', 'upper left'))
plt.figlegend(('Random', 'G_prim_users', 'DRL'))
plt.show()

# PLOT the results below for Percentage violated users
plt.figure(2)
plt.title("QoS Violations")
plt.grid(b=True, which='major', axis='both')
plt.ylim(0, 1)
plt.grid(color='k', linestyle='-', linewidth=0.1)
loc, labels = plt.yticks()
plt.xlim(x[0], x[-1] + 1)
plt.xlabel("Number of Secondary Users")
plt.ylabel("Normalised QoS Violations")
plt.plot(x, mean_violations_vector_Random, 'r-p', label='Random')
# plt.plot(x, mean_value_accuracy_HS, '-m^', label='HS')
#plt.plot(x, mean_violations_vector_migC, 'k-D', label='migC')
plt.plot(x, mean_violations_vector_G_prim_users, 'g-.s', label='G_prim_users')
plt.plot(x, mean_violations_vector_QL, '-b^', label='DRL')
# plt.plot(x, mean_value_accuracy_distance, 'k-.s', label='LB')
# plt.figlegend(('Dist', 'HS', 'Energ', 'Random', 'upper left'))
plt.figlegend(('Random', 'G_prim_users', 'DRL'))
plt.show()

# PLOT the results below for number of handoffs
plt.figure(3)
plt.title("Number of handoffs")
plt.grid(b=True, which='major', axis='both')
# plt.ylim(0,1)
plt.grid(color='k', linestyle='-', linewidth=0.1)
loc, labels = plt.yticks()
plt.xlim(x[0], x[-1] + 1)
plt.xlabel("Number of Secondary Users")
plt.ylabel("Number of handoffs in %")
plt.plot(x, [100 * x for x in mean_HO_vector_Random], 'r-p', label='Random')
# plt.plot(x, mean_value_accuracy_HS, '-m^', label='HS')
#plt.plot(x, [100 * x for x in mean_HO_vector_migC], 'k-D', label='migC')
plt.plot(x, [100 * x for x in mean_HO_vector_G_prim_users], 'g-.s', label='G_prim_users')
plt.plot(x, [100 * x for x in mean_HO_vector_QL], '-b^', label='DRL')
# plt.plot(x, mean_value_accuracy_distance, 'k-.s', label='LB')
# plt.figlegend(('Dist', 'HS', 'Energ', 'Random', 'upper left'))
plt.figlegend(('Random', 'G_prim_users', 'DRL'))
plt.show()




