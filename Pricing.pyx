# pp_module.pyx
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION  # Disable deprecated NumPy API
import numpy as np

# Define any cimport statements here for optimization
cimport numpy as np

cimport cython




@cython.boundscheck(False)
@cython.wraparound(False)
# Define the Cython function
def get_split_values(int leaf_type, int depth):
    cdef np.ndarray[np.double_t, ndim=1] split_values
    split_values = np.empty(depth, dtype=np.double)
    cdef int rem_val = leaf_type

    for l in range(depth):
        split_values[l] = rem_val // 2**(depth-l-1)
        rem_val = rem_val - int(split_values[l] * 2**(depth-l-1))

    split_values = 1.0 - split_values
    return split_values





@cython.boundscheck(False)
@cython.wraparound(False)
def get_w_indices(int leaf, int d):
    cdef int d_temp, w_index
    cdef list w_indices_for_leaf = [0]
    
    for d_temp in range(1, d):
        values = np.arange(2**d_temp - 1, 2**d_temp - 1 + 2**d_temp)
        w_index = int(leaf // (2**(d - d_temp)))
        w_indices_for_leaf.append(values[w_index])
    
    return w_indices_for_leaf

@cython.boundscheck(False)
@cython.wraparound(False)
def get_affected_leaves(int node, list w_indices_for_leaves):
    cdef list affected_leaves = []
    cdef int leaf
    cdef list values
    
    for leaf in range(len(w_indices_for_leaves)):
        values = w_indices_for_leaves[leaf]
        if node in values:
            affected_leaves.append(leaf)
    
    return affected_leaves


@cython.boundscheck(False)
@cython.wraparound(False)
def get_misclassification(np.ndarray[np.int32_t, ndim=2] x, np.ndarray[np.int32_t, ndim=2] y_k, int[:] leaf_pattern, int leaf_type) -> (double, int) :
    cdef int depth = len(leaf_pattern)

    cdef np.ndarray[np.double_t, ndim=1] split_values = np.zeros(depth, dtype=np.float64)
    cdef np.ndarray mask
    mask = np.ones(x.shape[0], dtype=bool)
    cdef np.ndarray[np.int32_t, ndim=2] x_filt = x.copy()
    cdef np.ndarray[np.int32_t, ndim=2] y_filt = y_k.copy()

    split_values=get_split_values(leaf_type,depth)

    for l in range(depth):
        mask &=  (x[:, leaf_pattern[l]] == split_values[l])

    # Apply the mask to filter x_filt and y_filt
    x_filt = x[mask]
    y_filt = y_k[mask]


    cdef int num_samples = len(y_filt)
    cdef double max_class_sum = 0.0

    if num_samples > 0:
        max_class_sum = np.max(np.sum(y_filt, axis=0))

    # return mask also
    if num_samples == 0:
        return (0.0,0,mask)  # Return 0 if y_filt is empty
    else:
        return (num_samples - max_class_sum, num_samples,mask)

# Add the optimized get_misclassification_bag function
@cython.boundscheck(False)
@cython.wraparound(False)
def get_misclassification_bag(np.ndarray[np.int32_t, ndim=2] x, np.ndarray[np.int32_t, ndim=2] y_k, tuple[list] leaf_pattern, int leaf_type) -> (double, int) :
    
    cdef int depth = len(leaf_pattern[0])+ len(leaf_pattern[1])
    cdef np.ndarray[np.double_t, ndim=1] split_values = np.zeros(depth, dtype=np.float64)
    cdef np.ndarray mask
    mask = np.ones(x.shape[0], dtype=bool)
    cdef np.ndarray[np.int32_t, ndim=2] x_filt = x.copy()
    cdef np.ndarray[np.int32_t, ndim=2] y_filt = y_k.copy()

    split_values=get_split_values(leaf_type,depth)


    for split in set(split_values):
        for feat in leaf_pattern[1-int(split)]: # we have 0 splits first
            mask &=  (x[:, int(feat)] == split)

    # Apply the mask to filter x_filt and y_filt
    x_filt = x[mask]
    y_filt = y_k[mask]


    cdef int num_samples = len(y_filt)
    cdef double max_class_sum = 0.0

    if num_samples > 0:
        max_class_sum = np.max(np.sum(y_filt, axis=0))

    # return mask also
    if num_samples == 0:
        return (0.0,0,mask)  # Return 0 if y_filt is empty
    else:
        return (num_samples - max_class_sum, num_samples,mask)


@cython.boundscheck(False)
@cython.wraparound(False)
# returns a pair of missclassification values
def get_misclassification_bag_sensitive(np.ndarray[np.int32_t, ndim=2] x, np.ndarray[np.int32_t, ndim=2] y_k, tuple[list] leaf_pattern, int leaf_type, list protected_features) :
    
    cdef int depth = len(leaf_pattern[0])+ len(leaf_pattern[1])
    cdef np.ndarray[np.double_t, ndim=1] split_values = np.zeros(depth, dtype=np.float64)
    cdef np.ndarray mask_sensitive
    cdef np.ndarray mask_non_sensitive
    mask_sensitive = np.ones(x.shape[0], dtype=bool)
    mask_non_sensitive = np.ones(x.shape[0], dtype=bool)
    cdef np.ndarray[np.int32_t, ndim=2] x_filt = x.copy()
    cdef np.ndarray[np.int32_t, ndim=2] y_filt = y_k.copy()

    split_values=get_split_values(leaf_type,depth)

    ## fill sensitive mask and non-sensitive mask
    for feat in protected_features: # sensitive
        mask_sensitive &=  (x[:, int(feat)] == 1)
        mask_non_sensitive &=  (x[:, int(feat)] == 0)

    
    for split in set(split_values):
        for feat in leaf_pattern[1-int(split)]: # we have 0 splits first
            mask_sensitive &=  (x[:, int(feat)] == split)
            mask_non_sensitive &=  (x[:, int(feat)] == split)
    
    # Apply the mask to filter x_filt and y_filt
    x_filt_sensitive = x[np.where(mask_sensitive)]
    y_filt_sensitive = y_k[np.where(mask_sensitive)]

    cdef int num_samples_sensitive = len(y_filt_sensitive)
    cdef double max_class_sum_sensitive = 0.0

    if num_samples_sensitive > 0:
        max_class_sum_sensitive = np.max(np.sum(y_filt_sensitive, axis=0))

     # Apply the mask to filter x_filt and y_filt

    x_filt_non_sensitive = x[np.where(mask_non_sensitive)]
    y_filt_non_sensitive = y_k[np.where(mask_non_sensitive)]

    cdef int num_samples_non_sensitive = len(y_filt_non_sensitive)
    cdef double max_class_sum_non_sensitive = 0.0

    if num_samples_sensitive > 0:
        max_class_sum_non_sensitive = np.max(np.sum(y_filt_non_sensitive, axis=0))

    return ((num_samples_sensitive - max_class_sum_sensitive,num_samples_sensitive),(num_samples_non_sensitive - max_class_sum_non_sensitive,num_samples_non_sensitive))


# returns a pair of missclassification values
def get_misclassification_bag_sensitive_class(np.ndarray[np.int32_t, ndim=2] x, np.ndarray[np.int32_t, ndim=2] y_k, tuple[list] leaf_pattern, int leaf_type, list protected_features,classOfInterest) :
    
    cdef int depth = len(leaf_pattern[0])+ len(leaf_pattern[1])
    cdef np.ndarray[np.double_t, ndim=1] split_values = np.zeros(depth, dtype=np.float64)
    cdef np.ndarray mask_sensitive
    cdef np.ndarray mask_non_sensitive
    mask_sensitive = y_k[:,classOfInterest].flatten()#np.ones(x.shape[0], dtype=bool)
    mask_non_sensitive = y_k[:,classOfInterest].flatten() #np.ones(x.shape[0], dtype=bool)
    cdef np.ndarray[np.int32_t, ndim=2] x_filt = x.copy()
    cdef np.ndarray[np.int32_t, ndim=2] y_filt = y_k.copy()

    split_values=get_split_values(leaf_type,depth)

    ## fill sensitive mask and non-sensitive mask
    for feat in protected_features: # sensitive
        mask_sensitive &=  (x[:, int(feat)] == 1)
        mask_non_sensitive &=  (x[:, int(feat)] == 0)

    for split in set(split_values):
        for feat in leaf_pattern[1-int(split)]: # we have 0 splits first
            mask_sensitive &=  (x[:, int(feat)] == split)
            mask_non_sensitive &=  (x[:, int(feat)] == split)
    
    # Apply the mask to filter x_filt and y_filt
    x_filt_sensitive = x[np.where(mask_sensitive)]
    y_filt_sensitive = y_k[np.where(mask_sensitive)]

    cdef int num_samples_sensitive = len(y_filt_sensitive)
    cdef double max_class_sum_sensitive = 0.0

    if num_samples_sensitive > 0:
        max_class_sum_sensitive = np.max(np.sum(y_filt_sensitive, axis=0))

     # Apply the mask to filter x_filt and y_filt

    x_filt_non_sensitive = x[np.where(mask_non_sensitive)]
    y_filt_non_sensitive = y_k[np.where(mask_non_sensitive)]

    cdef int num_samples_non_sensitive = len(y_filt_non_sensitive)
    cdef double max_class_sum_non_sensitive = 0.0

    if num_samples_sensitive > 0:
        max_class_sum_non_sensitive = np.max(np.sum(y_filt_non_sensitive, axis=0))

    return ((num_samples_sensitive - max_class_sum_sensitive,num_samples_sensitive),(num_samples_non_sensitive - max_class_sum_non_sensitive,num_samples_non_sensitive))

# Define the class with cdef
cdef class PP_node_fairness:
    # cdef public double[:, ::1] x

    cdef public int leaf, d_current, d, p,min_bucket_leaf
    cdef public list hist
    cdef public int status
    cdef public PP_node_fairness parent
    cdef public double split
    cdef public int[:] indices
    cdef public double LB
    cdef public double dual_variable_component
    cdef public double miss_component
    cdef public double fairness_component
    cdef public x
    cdef public y
    cdef public y_k
    cdef public pi
    cdef public gamma
    cdef public children
    cdef public level_nums
    cdef public branching_history
    cdef public myopic_choice
    
    cdef public sensitive_features
    cdef public mask_sensitive
    cdef public mask_non_sensitive

    # np.ndarray[np.int32_t, ndim=2] x
    # np.ndarray[np.int32_t, ndim=1] y, 
    # np.ndarray[np.int32_t, ndim=2] y_k
    # np.ndarray[np.float64_t, ndim=2] pi


    def __init__(self, int leaf, np.ndarray[np.int32_t, ndim=2] x, np.ndarray[np.int32_t, ndim=1] y, np.ndarray[np.int32_t, ndim=2] y_k, int d_current, int d, int p, np.ndarray[np.float64_t, ndim=2] pi,np.ndarray[np.float64_t, ndim=1] gamma,sensitive_features, np.ndarray[np.int32_t, ndim=2] branching_history, int myopic_choice,PP_node_fairness parent=None, list hist=[], int min_bucket_leaf=0):
        
        self.d = d
        self.min_bucket_leaf=min_bucket_leaf
        self.leaf = leaf
        self.x = x
        self.y = y
        self.y_k = y_k
        self.pi = pi
        self.gamma=gamma

        #self.mask_sensitive=mask_sensitive
        #self.mask_non_sensitive=mask_non_sensitive
        self.sensitive_features=sensitive_features
        self.get_fairness_masks()

        self.d_current = d_current
        self.p = p
        self.hist = hist
        self.status = 1  # active
        self.parent = parent
        self.split = get_split_values(leaf, d)[d_current]
        self.branching_history=branching_history
        self.myopic_choice=myopic_choice






        #### node evaluation comes after
        self.get_LB()

    @cython.boundscheck(False)
    @cython.wraparound(False)


    cdef double get_missLB_singular(self, int p_):
        cdef int leaf = self.leaf
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef np.ndarray[np.int32_t, ndim=1] y = self.y
        cdef int d_current = self.d_current
        cdef int p = self.p  # p is an int
        cdef int d = self.d
        cdef int min_bucket_leaf=self.min_bucket_leaf
        cdef int myopic_choice = self.myopic_choice
        cdef double split = self.split
        cdef np.ndarray[np.int32_t, ndim=2] branching_history = self.branching_history

        cdef float gini
        cdef float RHS_temp
        cdef int feature_temp
        cdef int leaf_temp
        cdef int level_temp
        cdef affected_leaves
        cdef int condition
        cdef int w_index
        cdef list w_indices_for_leaves


        
        ## Convert p_ to int for slicing
        cdef int p_index = <int>p_

        cdef int num_samples = x.shape[0]
        cdef int mask_count = 0
        cdef int count = 0

        for i in range(num_samples):
            if float(x[i, p_index]) == split:
                mask_count += 1
                count += int(y[i])

        
        if mask_count<=min_bucket_leaf:
            return 0 ## 0 count means invalid status!
        
        elif myopic_choice == 0:
            return min(((mask_count - count) / num_samples, 1.0 - (mask_count - count) / num_samples))
        
        else:
            gini = 1.0 - (count / mask_count) ** 2 #### check later
            return gini



    cdef double get_missLB_singular_sensitive(self, int p_):
        cdef int leaf = self.leaf
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef np.ndarray[np.int32_t, ndim=1] y = self.y
        cdef mask_sensitive=self.mask_sensitive
        cdef int d_current = self.d_current
        cdef int p = self.p  # p is an int
        cdef int d = self.d
        cdef int min_bucket_leaf=self.min_bucket_leaf
        cdef int myopic_choice = self.myopic_choice
        cdef double split = self.split
        cdef np.ndarray[np.int32_t, ndim=2] branching_history = self.branching_history

        cdef float gini
        cdef float RHS_temp
        cdef int feature_temp
        cdef int leaf_temp
        cdef int level_temp
        cdef affected_leaves
        cdef int condition
        cdef int w_index
        cdef list w_indices_for_leaves


        
        ## Convert p_ to int for slicing
        cdef int p_index = <int>p_

        cdef int num_samples = len(mask_sensitive)
        cdef int mask_count = 0
        cdef int count = 0

        for i in mask_sensitive:
            if float(x[i, p_index]) == split:
                mask_count += 1
                count += int(y[i])

        
        if mask_count<=min_bucket_leaf:
            return 0 ## 0 count means invalid status!
        
        elif myopic_choice == 0:
            return min(((mask_count - count) / num_samples, 1.0 - (mask_count - count) / num_samples))
        
        else:
            gini = 1.0 - (count / mask_count) ** 2 #### check later
            return gini

    cdef double get_missLB_singular_non_sensitive(self, int p_):
        cdef int leaf = self.leaf
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef np.ndarray[np.int32_t, ndim=1] y = self.y
        cdef mask_non_sensitive=self.mask_non_sensitive
        cdef int d_current = self.d_current
        cdef int p = self.p  # p is an int
        cdef int d = self.d
        cdef int min_bucket_leaf=self.min_bucket_leaf
        cdef int myopic_choice = self.myopic_choice
        cdef double split = self.split
        cdef np.ndarray[np.int32_t, ndim=2] branching_history = self.branching_history

        cdef float gini
        cdef float RHS_temp
        cdef int feature_temp
        cdef int leaf_temp
        cdef int level_temp
        cdef affected_leaves
        cdef int condition
        cdef int w_index
        cdef list w_indices_for_leaves


        
        ## Convert p_ to int for slicing
        cdef int p_index = <int>p_

        cdef int num_samples = len(mask_non_sensitive)
        cdef int mask_count = 0
        cdef int count = 0

        for i in mask_non_sensitive:
            if float(x[i, p_index]) == split:
                mask_count += 1
                count += int(y[i])

        
        if mask_count<=min_bucket_leaf:
            return 0 ## 0 count means invalid status!
        
        elif myopic_choice == 0:
            return min(((mask_count - count) / num_samples, 1.0 - (mask_count - count) / num_samples))
        
        else:
            gini = 1.0 - (count / mask_count) ** 2 #### check later
            return gini
    
  
    cdef void get_LB(self):
        cdef int d = self.d
        cdef int leaf = self.leaf
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef int d_current = self.d_current
        cdef np.ndarray[np.float64_t, ndim=2] pi = self.pi
        cdef np.ndarray[np.float64_t, ndim=1] gamma = self.gamma
        cdef double miss_component
        cdef int split = <int> self.split
        

        miss_component = self.get_missLB_singular(self.hist[1-split][-1]) 
        dual_variable_component =  pi[1-split][self.hist[1-split][-1]] if self.parent is None else pi[1-split][self.hist[1-split][-1]] + self.parent.dual_variable_component 
        fairness_component=gamma[0]*self.get_missLB_singular_sensitive(self.hist[1-split][-1]) + gamma[1]*self.get_missLB_singular_non_sensitive(self.hist[1-split][-1])
        self.dual_variable_component = dual_variable_component
        self.miss_component=miss_component
        self.fairness_component=fairness_component

        LB = miss_component + dual_variable_component + fairness_component
        self.LB = LB
        
    cdef public return_LB(self):
        cdef float LB
        LB=self.LB

        return LB

    cdef void get_fairness_masks(self):
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef int num_samples = x.shape[0]
        cdef sensitive_features = self.sensitive_features

        # Use a list to store indices that match the condition
        matching_indices_sensitive = []
        matching_indices_non_sensitive = []
    
        for i in range(num_samples):
            flag=0
            for feat in sensitive_features:
                if x[i, feat] == 1:
                    flag=1 #update flag value
            if flag==1:
                matching_indices_sensitive.append(i)
            else:
                matching_indices_non_sensitive.append(i) 

        # Convert the list of indices to a NumPy array
        self.mask_sensitive = matching_indices_sensitive
        self.mask_non_sensitive= matching_indices_non_sensitive




    def __repr__(self):
        representation = (self.hist, self.LB)
        return str(representation)

   
    cpdef public void generate_children(self):
        #int leaf, np.ndarray[np.int32_t, ndim=2] x, np.ndarray[np.int32_t, ndim=1] y, np.ndarray[np.int32_t, ndim=2] y_k, int d_current, int d, int p, np.ndarray[np.float64_t, ndim=2] pi, np.ndarray[np.int32_t, ndim=2] branching_history, int myopic_choice,PP_node parent=None, list hist=[]
        cdef list children = []
        cdef int p = self.p
        cdef int min_bucket_leaf=self.min_bucket_leaf
        cdef double split = self.split
        cdef int leaf = self.leaf
        cdef int d_current = self.d_current
        cdef int d = self.d
        cdef int split_int = <int> self.split
        cdef int next_split= <int> get_split_values(leaf, d)[d_current+1]
        
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef np.ndarray[np.int32_t, ndim=1] y = self.y
        cdef np.ndarray[np.int32_t, ndim=2] y_k = self.y_k
        cdef sensitive_features=self.sensitive_features
        cdef np.ndarray[np.float64_t, ndim=2] pi = self.pi
        cdef np.ndarray[np.float64_t, ndim=1] gamma = self.gamma


        cdef list hist = self.hist
        cdef np.ndarray[np.int32_t, ndim=2] branching_history=self.branching_history
        cdef int myopic_choice=self.myopic_choice
        

        #update x and y to include this split

        mask =  (x[:,hist[1-split_int][-1] ] == split)


        
        # Apply the mask to filter x_filt and y_filt
        x_filt = x[mask]
        y_filt = y[mask]
        y_k_filt = y_k[mask]

        ## fairness masks are updated within the node

        for feature in range(p):
            hist_copy=[hist[0][:],hist[1][:]] # copy shallowly
            
            hist_copy[1-next_split]+= [feature] # modify history

            children.append(PP_node_fairness(leaf, x_filt, y_filt, y_k_filt, d_current + 1, d, p, pi,gamma,sensitive_features, branching_history,myopic_choice,parent=self, hist=hist_copy ,min_bucket_leaf=min_bucket_leaf))

        self.children = children

    cpdef public void eval_node(self):


        if self.miss_component >= 1:
            self.status = 0
        
        elif self.d== self.d_current+1:
            self.status = 1

        else:
            self.generate_children()

    # Define __reduce__ to make the class pickleable
    def __reduce__(self):
        # Return a tuple containing a callable and its arguments for unpickling
        return (self.__class__, (
            self.leaf, self.x, self.y, self.y_k, self.d_current, self.d, self.p, self.pi,
            self.parent, self.hist
        ))


# Define the class with cdef
cdef class PP_node:
    # cdef public double[:, ::1] x

    cdef public int leaf, d_current, d, p,min_bucket_leaf
    cdef public list hist
    cdef public int status
    cdef public PP_node parent
    cdef public double split
    cdef public int[:] indices
    cdef public double LB
    cdef public double dual_variable_component
    cdef public double miss_component
    cdef public x
    cdef public y
    cdef public y_k
    cdef public pi
    cdef public children
    cdef public level_nums
    cdef public branching_history
    cdef public myopic_choice
    


    # np.ndarray[np.int32_t, ndim=2] x
    # np.ndarray[np.int32_t, ndim=1] y, 
    # np.ndarray[np.int32_t, ndim=2] y_k
    # np.ndarray[np.float64_t, ndim=2] pi


    def __init__(self, int leaf, np.ndarray[np.int32_t, ndim=2] x, np.ndarray[np.int32_t, ndim=1] y, np.ndarray[np.int32_t, ndim=2] y_k, int d_current, int d, int p, np.ndarray[np.float64_t, ndim=2] pi, np.ndarray[np.int32_t, ndim=2] branching_history, int myopic_choice,PP_node parent=None, list hist=[], int min_bucket_leaf=0):
        
        self.d = d
        self.min_bucket_leaf=min_bucket_leaf
        self.leaf = leaf
        self.x = x
        self.y = y
        self.y_k = y_k
        self.pi = pi
        self.d_current = d_current
        self.p = p
        self.hist = hist
        self.status = 1  # active
        self.parent = parent
        self.split = get_split_values(leaf, d)[d_current]
        self.branching_history=branching_history
        self.myopic_choice=myopic_choice




        self.get_related_indices()



        #### node evaluation comes after
        self.get_LB()

    @cython.boundscheck(False)
    @cython.wraparound(False)


    cdef double get_missLB_singular(self, int p_):
        cdef int leaf = self.leaf
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef np.ndarray[np.int32_t, ndim=1] y = self.y
        cdef int d_current = self.d_current
        cdef int p = self.p  # p is an int
        cdef int d = self.d
        cdef int min_bucket_leaf=self.min_bucket_leaf
        cdef int myopic_choice = self.myopic_choice
        cdef double split = self.split
        cdef np.ndarray[np.int32_t, ndim=2] branching_history = self.branching_history

        cdef float gini
        cdef float RHS_temp
        cdef int feature_temp
        cdef int leaf_temp
        cdef int level_temp
        cdef affected_leaves
        cdef int condition
        cdef int w_index
        cdef list w_indices_for_leaves


        
        ## Convert p_ to int for slicing
        cdef int p_index = <int>p_
        
        # not needed anymore
        # #for cons_ in branching_history:
        #    #w_indices_for_leaves=[get_w_indices(l, d) for l in range(int(2**d))]
        #    # (level, feature, leaf, RHS) -- updated
        #    #(w_index,feature,RHS) -- updated
        #    #(feature,RHS) -- updated
        #    #(feature,leaf, RHS) -- updated
        #    
        #    feature_temp=cons_[0]
        #    RHS_temp=cons_[-1]
        #    leaf_temp=cons_[1]

        #    #condition= get_split_values(leaf_temp,d).astype(int)[int(2**d/(len(leaves))-1)] ## will return whether that node dictates a positive or negative split for the leaf
        #    ## the position of the leaf in the list signifies "depth", hence the split value can be derived
        #    for condition in set(get_split_values(leaf_temp,d).astype(int)):
        #        if feature_temp == p_index and RHS_temp == 0 and leaf ==leaf_temp :
        #            return 100000000000.0  # Return an incredibly large value to avoid being chosen
        #        elif feature_temp != p_index and RHS_temp == 1 and leaf ==leaf_temp:
        #            return 100000000000.0

        cdef int num_samples = x.shape[0]
        cdef int mask_count = 0
        cdef int count = 0

        for i in range(num_samples):
            if float(x[i, p_index]) == split:
                mask_count += 1
                count += int(y[i])

        
        if mask_count<=min_bucket_leaf:
            return 0 ## 0 count means invalid status!
        
        elif myopic_choice == 0:
            return min(((mask_count - count) / num_samples, 1.0 - (mask_count - count) / num_samples))
        
        else:
            gini = 1.0 - (count / mask_count) ** 2 #### check later
            return gini



     

    
  
    cdef void get_LB(self):
        cdef int d = self.d
        cdef int leaf = self.leaf
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef int d_current = self.d_current
        cdef np.ndarray[np.float64_t, ndim=2] pi = self.pi
        cdef double miss_component
        cdef int split = <int> self.split
        

        miss_component = self.get_missLB_singular(self.hist[1-split][-1]) 
        dual_variable_component =  pi[1-split][self.hist[1-split][-1]] if self.parent is None else pi[1-split][self.hist[1-split][-1]] + self.parent.dual_variable_component 
        
        self.dual_variable_component = dual_variable_component
        self.miss_component=miss_component

        LB = miss_component + dual_variable_component
        self.LB = LB
        
    cdef public return_LB(self):
        cdef float LB
        LB=self.LB

        return LB

    cdef void get_related_indices(self):
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef double split = self.split
        cdef int split_int = <int> self.split
        cdef int hist_last_element = self.hist[1-split_int][-1]  # Get the last element from hist
        cdef int num_samples = x.shape[0]
    
        # Use a list to store indices that match the condition
        matching_indices = []
    
        for i in range(num_samples):
            if x[i, hist_last_element] == split:
                matching_indices.append(i)
    
        # Convert the list of indices to a NumPy array
        self.indices = np.array(matching_indices, dtype=np.int32)




    def __repr__(self):
        representation = (self.hist, self.LB)
        return str(representation)

   
    cpdef public void generate_children(self):
        #int leaf, np.ndarray[np.int32_t, ndim=2] x, np.ndarray[np.int32_t, ndim=1] y, np.ndarray[np.int32_t, ndim=2] y_k, int d_current, int d, int p, np.ndarray[np.float64_t, ndim=2] pi, np.ndarray[np.int32_t, ndim=2] branching_history, int myopic_choice,PP_node parent=None, list hist=[]
        cdef list children = []
        cdef int p = self.p
        cdef int min_bucket_leaf=self.min_bucket_leaf
        cdef double split = self.split
        cdef int leaf = self.leaf
        cdef int d_current = self.d_current
        cdef int d = self.d
        cdef int split_int = <int> self.split
        cdef int next_split= <int> get_split_values(leaf, d)[d_current+1]
        
        cdef np.ndarray[np.int32_t, ndim=2] x = self.x
        cdef np.ndarray[np.int32_t, ndim=1] y = self.y
        cdef np.ndarray[np.int32_t, ndim=2] y_k = self.y_k
        cdef np.ndarray[np.float64_t, ndim=2] pi = self.pi
        cdef list hist = self.hist
        cdef np.ndarray[np.int32_t, ndim=2] branching_history=self.branching_history
        cdef int myopic_choice=self.myopic_choice
        

        #update x and y to include this split

        mask =  (x[:,hist[1-split_int][-1] ] == split)
        
        # Apply the mask to filter x_filt and y_filt
        x_filt = x[mask]
        y_filt = y[mask]
        y_k_filt = y_k[mask]

        
        for feature in range(p):
            hist_copy=[hist[0][:],hist[1][:]] # copy shallowly
            
            hist_copy[1-next_split]+= [feature] # modify history

            children.append(PP_node(leaf, x_filt, y_filt, y_k_filt, d_current + 1, d, p, pi, branching_history,myopic_choice,parent=self, hist=hist_copy ,min_bucket_leaf=min_bucket_leaf))

        self.children = children

    cpdef public void eval_node(self):


        if self.miss_component >= 1:
            self.status = 0
        
        elif self.d== self.d_current+1:
            self.status = 1

        else:
            self.generate_children()

    # Define __reduce__ to make the class pickleable
    def __reduce__(self):
        # Return a tuple containing a callable and its arguments for unpickling
        return (self.__class__, (
            self.leaf, self.x, self.y, self.y_k, self.d_current, self.d, self.p, self.pi,
            self.parent, self.hist
        ))