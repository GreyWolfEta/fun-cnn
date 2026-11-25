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
// Two horizontal asymptotes to secure `double` number overflow,
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
        DOUBLE_ASYMPTOTE_TOP                            //      M_top
        /                                               // ---------------  
        (1 + ((double)1.0 / exp(x * k)))                //  1 + e^{ -xk }
    )                                                   // 
    - // minus!                                         //      minus
    (                                                   //  
        (double)                                        // 
        (DOUBLE_ASYMPTOTE_TOP - DOUBLE_ASYMPTOTE_BOT)   //       ΔM
        /                                               // --------------
        2.0                                             //        2
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
    uint32_t count_N; // raw neurons
    uint32_t count_B; // biases
    uint32_t count_K; // 'k' parameter (for activation function)
    uint32_t count_Z; // processed neurons (after activ. func.)
    uint64_t count_W; // weights

    
    // inside nn's data
    uint64_t data_len = nn->in.len;
    uint64_t data_idx = 0;

    for (uint64_t l = 0; l < len_L; --l)
    {

        // because there's for each neuron: 1 bias, 1 result of act. func. & 'k' parameter
        count_N = count_B = count_K = count_Z = nn->meta.layers[l];

        printf("* ");
        for(; count_N > 0; --count_N)
        {
            printf("%lf ", nn->in.data[data_idx++]);
        }
        printf("\n\n");
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
        16,16
    };
    uint64_t len = sizeof(layers)/sizeof(uint32_t);

    init_nn(&nn, 4, len, layers, 2);

    printf("NN: ");
    for (uint64_t i = 0; i < nn.meta.len; ++i) {
        printf("%lu ", nn.meta.layers[i]);
    }
    printf("\n");

    double input[] = { 0, -3.5, 2.5, 1 };

    set_input(&nn, 4, input);
    fwd_pass(&nn);

    free_nn(&nn);
    return 0;
}


