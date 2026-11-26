#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <time.h>




/*
 * Fun Neural Network In C
 * ~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This struct contains all the infromation about the neural network:
 *
 *  `meta`  "metadata"  information about its layer composition
 *  `in`    "inside"    its actual data (neurons, weights & biases)
 *
 * 
 */
typedef struct {
    struct {
        uint64_t  len;
        uint32_t* layers;
    } meta;
    struct {
        uint64_t len;
        double* data;
    } in;
} NN;



// ──────────────────────────────────── MATH

double random_double(double min, double max)
{
    double range = max - min;
    double div = RAND_MAX / range;
    return min + (rand() / div);
}


// 
// Two horizontal asymptotes to make it impossible for `double` to overflow,
// basically it computes in "float", but it is double, just limited
//
// ps: I really hope my math won't fail me...
//
static const double DOUBLE_ASYMPTOTE_TOP =  DBL_MAX / 2.0;
static const double DOUBLE_ASYMPTOTE_BOT = -DBL_MAX / 2.0;


// 
// Activation Function
// ~~~~~~~~~~~~~~~~~~~
//  a1 = s(w * a0 + b)  ← this 's'
//
// Follows equation:
// 
//  y = M_top / ( 1 + 1 / e^{ x * k } ) - (M_top - M_bot) / 2
//
// where:
//  - M are asymtotes ("maximum" and "minimum")
//  - x is neuron
//  - k is a kind of bias for this act. func.
//
double activation_function(double x, double k)
{
    return ((
        (double)
        DOUBLE_ASYMPTOTE_TOP                            //        M_top
        /                                               // -------------------  
        (1 + ((double)1.0 / exp(x * k)))                //  1 +  1 / e^{ -xk }
    )                                                   // 
    - // minus!                                         //      minus
    (                                                   //  
        (double)                                        // 
        (DOUBLE_ASYMPTOTE_TOP - DOUBLE_ASYMPTOTE_BOT)   //   ΔM
        /                                               // ------
        2.0                                             //    2
    )
    );
}

// 
// Partial differrentiation of activation function with respect to 'x':
//
//  dy/dx = M * (??)
//
double activation_function_dx(double x, double k)
{
    // @todo: implement ds/dx
    double e = exp(k * x);
    return DOUBLE_ASYMPTOTE_TOP * (
            (double)(k * e)
            /
            pow((1 + e), 2)
            );
}

// 
// Partial differrentiation of activation function with respect to 'k':
//
//  dy/dx = M * (??)
//
double activation_function_dk(double x, double k)
{
    // @todo: implement ds/dk
}


// ──────────────────────────────────── NN FUNCTIONALITY

bool init_nn(NN* nn, uint32_t input, const uint64_t len, uint32_t* layers, uint32_t output)
{
    uint64_t data_len = 0;
    uint64_t layer_count = 2;

    data_len += input;
    for (uint64_t i = 0; i < len; ++i, ++layer_count)
    {
        if (i == 0)
        {
            data_len += input * layers[i] + layers[i] * 3;
        }
        else
        {
            data_len += layers[i - 1] * layers[i] + layers[i] * 3;
        }
    }
    data_len += len > 0 ? layers[len - 1] * output + output * 3: output * 3;
    
    printf("allocating %zu of uint32\n", layer_count);
    nn->meta.len = layer_count;
    nn->meta.layers = malloc(sizeof(uint32_t) * layer_count);

    printf("allocating %zu of doubles\n", data_len);
    nn->in.len = data_len;
    nn->in.data = malloc(sizeof(double) * data_len);

    if (!nn->meta.layers || !nn->in.data)
    {
        fprintf(stderr, "Error! Cannot allocate enough memory\n");
        return 0;
    }

    printf("allocated, setting the neural network\n");
    for (uint64_t i = 0; i < len; ++i)
    {
        printf("L:%lu (%lu)\n", layers[i], i);
        nn->meta.layers[i+1] = layers[i];
    }
    nn->meta.layers[0] = input;
    nn->meta.layers[nn->meta.len - 1] = output;

    printf("randomizing nn\n");
    for (uint64_t i = 0; i < nn->in.len; ++i)
    {
        nn->in.data[i] = random_double(-DBL_MIN/2, DBL_MAX/2);
    }
    return nn;
}


void free_nn(NN* nn)
{
    // free allocated data
    // - maybe i should add some safety NULL-check, but not now (i trust current myself) 
    free(nn->meta.layers);
    free(nn->in.data);
}




// ────────────────────────────────────

// set input layer
void set_input(NN* nn, uint64_t len, double* data)
{
    if (nn->meta.layers[0] != len)
    {
        fprintf(stderr, "set_input: error: input.len != actual number of input neurons\n");
        return;
    }
    for (uint64_t i = 0; i < len; ++i)
    {
        // `nn->in.data` starts with first input layer, therefore this is ok
        nn->in.data[i] = data[i];
    }
}

// Foward pass
void fwd_pass(NN* nn)
{
    uint64_t len_L = nn->meta.len; // number of layers


    // 
    // the memory could be optimalized here,
    // i think that only one counter would be
    // enough, but for now, i will keep it like that...
    //
    uint32_t count_N0;  // raw neurons
    uint32_t count_N1;  // biases
                        // 'k' parameter (for activation function)
                        // processed neurons (after activ. func.)
                        // weights

    
    // inside nn's data
    uint64_t data_len = nn->in.len;
    uint64_t idx_base = 0;

    for (uint64_t l = 0; l < len_L - 1; --l)
    {

        // because there's for each neuron: 1 bias, 1 result of act. func. & 'k' parameter
        count_N0 = nn->meta.layers[l];
        count_N1 = nn->meta.layers[l + 1];

        uint64_t idx_n0 = idx_base;                             // index of actual neuron layer
        uint64_t idx_w  = idx_base + count_N0;                  // index of weights
        uint64_t idx_b  = idx_w + count_N0 * count_N1;          // index of biases
        uint64_t idx_n1 = idx_w + (2 * count_N1);               // index of the next pre-activated neuron layer
                                                                //          (before activation_function)

        // 
        // preparing next neuron layer for sum of values from previous layer
        //
        for (int n1 = 0; n1 < count_N1 && n1 < data_len; ++n1)
        {
            nn->in.data[n1 + idx_n1] = 0;
        }


        // 
        // ===========================
        // MATRIX N→W→N CALCULATION
        // ===========================
        //
        // previous layer's neuron
        for (int n0 = 0; n0 < count_N0 && n0 + idx_n0 < data_len; ++n0)
        {
            // next layer's neuron
            for (int n1 = 0; n1 < count_N1 && n1 + idx_n1 < data_len; ++n1)
            {
                // -------------------------------------------------------------------------------------------  next target layer of neurons, these neurons store values before applying activ. func.
                // |                         -----------------------------------------------------------------  value of the previous layer (values from activation function)
                // |                         |                         ---------------------------------------  weights between these two layers
                // |                         |                         |       -------------------------------  target pre-activated neuron in the next layer
                // |                         |                         |       |    --------------------------  points to the first weight stored in this "weight layer"
                // |                         |                         |       |    |          ---------------  shifts pointer of weights to group of weights of the appropriate neuron
                // |                         |                         |       |    |          |      
                // V                         V                         V       V    V          V      
                // pre-activation layer  +=  prevous activated layer * weights[n1 + base_idx + offset]
                nn->in.data[n1 + idx_n1] += nn->in.data[n0 + idx_n0] * nn->in.data[n1 + idx_w + n1 * count_N1];
            }
        }

        // 
        // ===========================================
        // ADD BIASES AND APPLYING ACTIVATION FUNCTION
        // ===========================================
        //
        for (int n1 = 0; n1 + idx_n1 < count_N1; ++n1)
        {
            // @todo: implement 'b' and 'k' biases with application of activation function
        }
    }
}

// Retrieve data from ouput layer in NN and apply function to make it probabilistic
void rtr_data(NN* nn, uint64_t bufsize, double* dst)
{
    // @todo: softmax, output data retrieve
}




// ──────────────────────────────────── MAIN CODE

int main( void)
{
    srand(time(NULL));

    NN nn;
    uint32_t layers[] = {
        2, 2
    };
    uint64_t len = sizeof(layers)/sizeof(uint32_t);

    init_nn(&nn, 1, len, layers, 2);

    printf("NN: ");
    for (uint64_t i = 0; i < nn.meta.len; ++i) {
        printf("%lu ", nn.meta.layers[i]);
    }
    printf("\n");

    double input[] = { 0, -3.5, 2.5, 1 };

    set_input(&nn, 2, input);
    fwd_pass(&nn);

    free_nn(&nn);
    return 0;
}


