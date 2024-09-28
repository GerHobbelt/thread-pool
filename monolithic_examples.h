
#pragma once

#if defined(BUILD_MONOLITHIC)

#ifdef __cplusplus
extern "C" {
#endif

int bs_threadpool_light_test_main(void);
int bs_threadpool_test_main(void);
int bs_threadpool_maniacal_test_main(void);

#ifdef __cplusplus
}
#endif

#endif
