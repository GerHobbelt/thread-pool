
#define BUILD_MONOLITHIC 1
#include "monolithic_examples.h"

#define USAGE_NAME   "threadpool"

#include "monolithic_main_internal_defs.h"

MONOLITHIC_CMD_TABLE_START()
	{ "light_test", {.fpp = bs_threadpool_light_test_main } },
	{ "test", {.fpp = bs_threadpool_test_main } },
	{ "maniac", {.fpp = bs_threadpool_maniacal_test_main } },
MONOLITHIC_CMD_TABLE_END();

#include "monolithic_main_tpl.h"
