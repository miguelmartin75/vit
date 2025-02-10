#include <catch2/catch_test_macros.hpp>

#include "../vit.h"

TEST_CASE( "shape literal", "[shape]" ) {
    Shape shape = shape_lit({-1, 16, 16});
    REQUIRE(shape_nelem(shape) == -256);

    Shape ref = shape_lit({3, 224, 224});
    shape_autofill(&shape, ref);
    REQUIRE(shape.dims[0] == 588);
}

TEST_CASE("init", "[tensor]") {
    Tensor zeros = tensor_zeros(shape_lit({3, 224, 224}));
    REQUIRE(shape_nelem(zeros.shape) == 150528);

    Tensor new_view = tensor_view(zeros, shape_lit({-1, 16*16}));
    REQUIRE(shape_nelem(new_view.shape) == 150528);
    REQUIRE(new_view.shape.dims[0] == 588);
}

TEST_CASE( "binary math ops", "[tensor]" ) {
}
